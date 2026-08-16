import adi
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import fftconvolve

sample_rate = 56e6
sample_time = 1.0 / float(sample_rate)
f_center = 2600e6
block_size = 16384
ramp_time = sample_time * block_size
max_freq = sample_rate
c = 3e8  # speed of light
zero_pad_factor = 4

RANGE_MIN = 0.0
RANGE_MAX = 100.0
DB_MIN = -80.0
DB_MAX = 80.0
WATERFALL_DEPTH = 200
SIGNAL_DISPLAY_SAMPLES = 2000

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
valid = freqs >= 0
range_axis = range_axis[valid]
visible = (range_axis >= RANGE_MIN) & (range_axis <= RANGE_MAX)
range_axis = range_axis[visible]
range_step = range_axis[1] - range_axis[0]


def compute_range(data):
    windowed = data * window

    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))
    mag_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

    return mag_db[valid][visible]


def find_roll(received):
    """Return the shift that aligns received samples to the TX waveform."""
    rx_scale = np.max(np.abs(received))
    if rx_scale == 0:
        return 0

    rx = received / rx_scale
    correlation = fftconvolve(rx, alignment_kernel, mode="full")
    lag = np.argmax(np.abs(correlation)) - (len(received) - 1)
    return -lag


# Setup SDR
sdr = adi.Pluto("ip:192.168.2.1")

# TX
sdr.tx_rf_bandwidth = int(sample_rate)
sdr.tx_lo = int(f_center)
sdr.tx_hardwaregain_chan0 = 0

# RX
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(f_center)
sdr.rx_hardwaregain_chan0 = 70
sdr.rx_buffer_size = block_size

sdr.sample_rate = int(sample_rate)

sdr.tx_cyclic_buffer = True
sdr.tx(sawtooth * (2**14))

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
    received_samples = sdr.rx() / (2**11)
    received_samples = np.roll(received_samples, find_roll(received_samples))
    mixed_product = np.conj(sawtooth) * received_samples
    mag_db = compute_range(mixed_product)
    curve.setData(range_axis, mag_db)

    history[:-1, :] = history[1:, :]
    history[-1, :] = mag_db
    img.setImage(history, autoLevels=False, levels=(DB_MIN, DB_MAX))


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)

app.exec()

sdr.tx_destroy_buffer()
