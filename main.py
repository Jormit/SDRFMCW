import adi
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import fftconvolve

sample_rate = 56e6
sample_time = 1.0 / float(sample_rate)
f_center = 2500e6
block_size = 16384
ramp_time = sample_time * block_size
max_freq = sample_rate
c = 3e8  # speed of light
zero_pad_factor = 4

RANGE_MIN = 0  # set to a number (e.g. 0.0) later to clip; None = show everything
RANGE_MAX = 50  # set to a number (e.g. 100.0) later to clip; None = show everything
# These now describe the post-background-subtraction display range (dB above
# the captured background), not absolute dB. Tune once you see real data.
DB_MIN = -10.0
DB_MAX = 70.0
WATERFALL_DEPTH = 200
SIGNAL_DISPLAY_SAMPLES = 2000
N_BACKGROUND_FRAMES = 20  # frames averaged to build the background estimate
N_CHIRP_AVERAGE = 8  # chirps averaged (in power domain) per displayed frame

# Generate the Sawtooth
bandwidth = max_freq
slope = bandwidth / ramp_time
f0 = -bandwidth / 2

times = np.linspace(0, ramp_time, block_size)
phase = 2 * np.pi * (f0 * times + 0.5 * slope * times**2)
sawtooth = np.exp(-1j * phase)
alignment_kernel = np.conj(sawtooth[::-1])
window = np.hanning(block_size)


n_fft = block_size * zero_pad_factor
freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=sample_time))
range_axis = freqs * c / (2 * slope)

# Show the full spectrum (both +/- range bins) unless RANGE_MIN/MAX are set above.
valid = np.ones_like(freqs, dtype=bool)
range_axis = range_axis[valid]
if RANGE_MIN is None:
    RANGE_MIN = range_axis.min()
if RANGE_MAX is None:
    RANGE_MAX = range_axis.max()
visible = (range_axis >= RANGE_MIN) & (range_axis <= RANGE_MAX)
range_axis = range_axis[visible]
range_step = range_axis[1] - range_axis[0]
print(
    f"Displaying full range axis: {range_axis.min():.1f} m to {range_axis.max():.1f} m"
)


def compute_range(data):
    windowed = data * window
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))
    mag_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
    return mag_db[valid][visible]


def find_roll(received):
    """Return the shift that aligns received samples to the TX waveform.
    Kept as a diagnostic: once TDD sync is working, this value should be
    stable (near-constant, small) frame to frame instead of jumping around.
    """
    rx_scale = np.max(np.abs(received))
    if rx_scale == 0:
        return 0
    rx = received / rx_scale
    correlation = fftconvolve(rx, alignment_kernel, mode="full")
    lag = np.argmax(np.abs(correlation)) - (len(received) - 1)
    return -lag


# ---------------------------------------------------------------------------
# Setup SDR
# ---------------------------------------------------------------------------
sdr_ip = "ip:192.168.2.1"
sdr = adi.Pluto(sdr_ip)
tddn = adi.tddn(sdr_ip)  # requires Pluto firmware >= 0.39 and pyadi-iio >= 0.18

# NOTE: only set sample_rate / rx_lo / tx_lo ONCE per power-up if you want
# a repeatable TX/RX phase relationship. If you re-run this script without
# power-cycling the Pluto, changing these again can break that alignment.
sdr.sample_rate = int(sample_rate)

# TX
sdr.tx_rf_bandwidth = int(sample_rate)
sdr.tx_lo = int(f_center)
sdr.tx_hardwaregain_chan0 = 0
sdr.tx_enabled_channels = [0]
sdr.tx_cyclic_buffer = True  # must be True to use TDD-gated transmit

# RX
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(f_center)
sdr.rx_hardwaregain_chan0 = 20
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = block_size
sdr._rxadc.set_kernel_buffers_count(1)  # don't let stale buffers queue up

# ---------------------------------------------------------------------------
# TDD engine configuration
# One TDD frame == one chirp period (block_size samples)
# ---------------------------------------------------------------------------
frame_length_ms = ramp_time * 1000.0

tddn.startup_delay_ms = 0
tddn.frame_length_ms = frame_length_ms
tddn.burst_count = 0  # 0 = repeat indefinitely

# RX DMA SYNC
tddn.channel[1].on_raw = 0
tddn.channel[1].off_raw = 10
tddn.channel[1].polarity = 0
tddn.channel[1].enable = 1

# TX DMA SYNC
tddn.channel[2].on_raw = 0
tddn.channel[2].off_raw = 10
tddn.channel[2].polarity = 0
tddn.channel[2].enable = 1

tddn.sync_external = False  # use the internal counter, no external trigger pin
tddn.enable = True

# Queue the chirp, then fire the sync so TX/RX DMA start together
sdr.tx(sawtooth * (2**14))
tddn.sync_soft = 1

# Sanity check: the quantized frame length may not exactly equal ramp_time.
# If it drifts noticeably from block_size samples, print a warning so you
# know to adjust block_size or accept the small discrepancy.
actual_frame_samples = tddn.frame_length_ms / 1000.0 * sample_rate
print(f"Requested frame length: {frame_length_ms:.6f} ms " f"({block_size} samples)")
print(
    f"Actual TDD frame length: {tddn.frame_length_ms:.6f} ms "
    f"({actual_frame_samples:.1f} samples), raw={tddn.frame_length_raw}"
)

# ---------------------------------------------------------------------------
# One-time range calibration.
# With TDD sync active, the TX/RX delay (cables + TDD reset overhead +
# ADC/DAC pipeline) is now a FIXED constant instead of jumping around.
# Measure it once here by averaging the matched-filter roll over several
# frames, then bake it in as a static correction. Do NOT recompute this
# every frame in update() -- that's what was flattening your display before,
# since it force-aligns whatever correlates best instead of showing the
# true, fixed feedthrough position.
# ---------------------------------------------------------------------------
N_CAL_FRAMES = 30
cal_rolls = []
for _ in range(N_CAL_FRAMES):
    cal_samples = sdr.rx() / (2**11)
    cal_rolls.append(find_roll(cal_samples))

cal_rolls = np.array(cal_rolls)
calibration_offset = int(np.median(cal_rolls))
print(
    f"Calibration rolls (samples): min={cal_rolls.min()} "
    f"max={cal_rolls.max()} median={calibration_offset}"
)
if cal_rolls.max() - cal_rolls.min() > 2:
    print(
        "WARNING: calibration roll is not stable frame-to-frame -- "
        "TDD sync may not be locked. Re-check firmware/config before "
        "trusting this calibration."
    )

# Approximate range this offset corresponds to, just for sanity-checking
# against what you saw in the plot before calibration:
cal_range_m = calibration_offset * (c / (2 * sample_rate))
print(
    f"Calibration offset corresponds to ~{cal_range_m:.2f} m -- "
    f"this is where your feedthrough peak was sitting."
)


def get_chirp_mag_db():
    """One TDD-synced chirp -> calibrated, dechirped range spectrum (dB)."""
    received_samples = sdr.rx() / (2**11)
    received_samples = np.roll(received_samples, calibration_offset)
    mixed_product = np.conj(sawtooth) * received_samples
    return compute_range(mixed_product)


def get_averaged_mag_db(n_chirps=N_CHIRP_AVERAGE):
    """Incoherently average n_chirps chirps (in power domain) to reduce the
    noise floor by roughly sqrt(n_chirps). This is magnitude-only averaging
    -- it does NOT preserve phase, so it helps SNR but is not a step toward
    Doppler/velocity processing (that needs coherent accumulation instead).
    """
    power_acc = np.zeros(len(range_axis))
    for _ in range(n_chirps):
        power_acc += 10 ** (get_chirp_mag_db() / 10.0)
    power_acc /= n_chirps
    return 10 * np.log10(power_acc + 1e-12)


# ---------------------------------------------------------------------------
# Background subtraction.
# Captures a static clutter/feedthrough map and subtracts it from every
# subsequent frame, so only CHANGES in the scene (moving/added targets)
# show up. Averages in linear power domain (not dB) before converting back,
# which is the statistically correct way to average noisy spectra.
# Call capture_background() again any time the scene needs to be re-zeroed
# (e.g. after moving the radar, or if clutter has changed) -- bound to the
# 'B' key in the GUI below.
# ---------------------------------------------------------------------------
def capture_background(n_frames=N_BACKGROUND_FRAMES):
    print(f"Capturing background over {n_frames} frames -- keep the scene clear...")
    power_acc = np.zeros(len(range_axis))
    for _ in range(n_frames):
        power_acc += 10 ** (get_chirp_mag_db() / 10.0)
    power_acc /= n_frames
    print("Background captured.")
    return 10 * np.log10(power_acc + 1e-12)


background = capture_background()

app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
win.setWindowTitle("SDR FMCW")
layout = QtWidgets.QVBoxLayout(win)
layout.setContentsMargins(0, 0, 0, 0)
layout.setSpacing(0)

plot = pg.PlotWidget(title="Range Spectrum")
layout.addWidget(plot, stretch=1)
curve = plot.plot(pen="y")
plot.setLabel("bottom", "Range", units="m")
plot.setLabel("left", "Magnitude", units="dB")
plot.setXRange(RANGE_MIN, RANGE_MAX, padding=0)
plot.setYRange(DB_MIN, DB_MAX, padding=0)

waterfall = pg.PlotWidget(title="Range Waterfall")
layout.addWidget(waterfall, stretch=2)
img = pg.ImageItem(axisOrder="row-major")
waterfall.addItem(img)
img.setLookupTable(pg.colormap.get("viridis").getLookupTable())
img.setImage(
    np.full((WATERFALL_DEPTH, len(range_axis)), DB_MIN),
    autoLevels=False,
    levels=(DB_MIN, DB_MAX),
    pos=(range_axis[0], 0),
    scale=(range_step, 1),
)
waterfall.setLabel("bottom", "Range", units="m")
waterfall.setLabel("left", "Scan")
waterfall.setXRange(RANGE_MIN, RANGE_MAX, padding=0)
waterfall.setYRange(0, WATERFALL_DEPTH, padding=0)

win.showMaximized()

history = np.full((WATERFALL_DEPTH, len(range_axis)), DB_MIN)


def update():
    mag_db = get_averaged_mag_db() - 0 * background
    curve.setData(range_axis, mag_db)

    history[:-1, :] = history[1:, :]
    history[-1, :] = mag_db
    img.setImage(history, autoLevels=False, levels=(DB_MIN, DB_MAX))


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)

app.exec()

tddn.enable = False
sdr.tx_destroy_buffer()
