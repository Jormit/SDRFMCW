import adi

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
import numpy as np
import matplotlib.pyplot as plt

sample_rate = 56e6
sample_time = 1.0/float(sample_rate)
f_center = 900e6
block_size = 16384
ramp_time = sample_time * block_size
max_freq = sample_rate/2
c = 3e8  # speed of light


def plot_sft(data, sample_time, block_size, title):
    plt.figure()
    # Peform STFT
    g_std = 40
    window_samples = 100
    w = gaussian(window_samples, std=g_std, sym=True)
    SFT = ShortTimeFFT(w, hop=10, fs=1/sample_time, scale_to='magnitude', fft_mode='centered')
    Sx = SFT.stft(data)

    # Plot Signal
    plt.imshow(np.abs(Sx), extent=SFT.extent(block_size), aspect='auto')
    plt.title(title)
    plt.colorbar()


def plot_range(data, sample_time, block_size, bandwidth, ramp_time, title="Range Spectrum"):
    slope = bandwidth / ramp_time  # Hz/s

    # Window to control sidelobes before the range FFT
    window = np.hanning(block_size)
    spectrum = np.fft.fftshift(np.fft.fft(data * window))
    freqs = np.fft.fftshift(np.fft.fftfreq(block_size, d=sample_time))

    ranges = freqs * c / (2 * slope)

    mag_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

    # Only positive ranges are physical (negative beat freq = negative range)
    valid = ranges >= 0
    ranges = ranges[valid]
    mag_db = mag_db[valid]

    max_unambig_range = (1/sample_time) * c / (4 * slope)  # Nyquist limit on f_beat = fs/2

    plt.figure()
    plt.plot(ranges, mag_db)
    plt.xlabel("Range (m)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"{title}  (max unambiguous range \u2248 {max_unambig_range:.1f} m)")
    plt.grid(True)
    plt.tight_layout()

    return ranges, mag_db


# Generate the Sawtooth
times = np.linspace(0, ramp_time, block_size)
freqs = np.linspace(0, max_freq, block_size)
sawtooth = np.exp(-np.pi*1j*times*freqs)

plot_sft(sawtooth, sample_time, block_size, "Transmit Signal")

# Setup SDR
sdr = adi.Pluto("ip:192.168.2.1")

#TX
sdr.tx_rf_bandwidth = int(sample_rate/2.0)
sdr.tx_lo = int(f_center)
sdr.tx_hardwaregain_chan0 = 0        # TX gain, in dB (range roughly -89.75 to 0)

#RX
sdr.rx_rf_bandwidth = int(sample_rate/2.0)  # filter cutoff, in Hz
sdr.rx_lo = int(f_center)
sdr.rx_hardwaregain_chan0 = 20         # RX gain, in dB (range roughly -3 to 70)
sdr.rx_buffer_size = block_size

# Shared sample rate
sdr.sample_rate = int(sample_rate)

# Enable cyclic buffer BEFORE calling tx()
sdr.tx_cyclic_buffer = True
sdr.tx(sawtooth * (2**14))

# Receive and plot
sdr.rx()
received_samples = sdr.rx() / (2**11)

def find_roll(reference, received):
    """
    Find the integer sample shift that best aligns `received` with `reference`
    via cross-correlation. Returns the roll amount to apply to `received`.
    """
    ref = reference / np.max(np.abs(reference))
    rx = received / np.max(np.abs(received))

    corr = np.correlate(rx, ref, mode='full')
    lag = np.argmax(np.abs(corr)) - (len(ref) - 1)

    return -lag

received_samples = np.roll(received_samples, find_roll(sawtooth, received_samples))

plot_sft(received_samples, sample_time, block_size, "Received Signal")


# Mixed Result
mixed_product = np.conj(sawtooth) * received_samples
plot_sft(mixed_product, sample_time, block_size, "After Mixer")

# Range spectrum
plot_range(mixed_product, sample_time, block_size, max_freq, ramp_time, "Range Spectrum")
#plt.xlim(0, 50)

sdr.tx_destroy_buffer()
plt.show()