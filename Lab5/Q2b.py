from scipy.io.wavfile import read, write
import numpy as np
import matplotlib.pyplot as plt

sample_rate, data = read('GraviteaTime.wav')

dat_1 = data[:, 0]
dat_2 = data[:, 1]

time = np.arange(len(dat_1)) / sample_rate

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(time, dat_1)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Channel 1')

plt.subplot(2, 1, 2)
plt.plot(time, dat_2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Channel 2')

plt.savefig('Original channel signal.png')
plt.tight_layout()
plt.show()

start = 0.02
end = 0.05

#must be integers or an error occurs due to use index method when zooming in.
start_location = int(start * sample_rate)
end_location = int(end * sample_rate)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time[start_location:end_location], dat_1[start_location:end_location])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Zoomed in Channel 1')

plt.subplot(2, 1, 2)
plt.plot(time[start_location:end_location], dat_2[start_location:end_location])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Zoomed in Channel 2')

plt.savefig('Zoomed channel signal.png')
plt.tight_layout()
plt.show()

#transform to freq
fft_1 = np.fft.fft(dat_1)
fft_2 = np.fft.fft(dat_2)
freqs = np.fft.fftfreq(len(fft_1), 1/sample_rate)

filtered_fft1 = fft_1.copy()
filtered_fft1[np.abs(freqs) > 880] = 0

filtered_fft2 = fft_2.copy()
filtered_fft2[np.abs(freqs) > 880] = 0

#back to time
dat_1_filtered = np.fft.ifft(filtered_fft1).real
dat_2_filtered = np.fft.ifft(filtered_fft2).real
plt.figure(figsize=(12, 10))

# === FIGURE 1: Fourier amplitudes for channel 0 ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(freqs[:len(dat_1)//2], np.abs(fft_1[:len(dat_1)//2]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Original FFT - Channel 1')
plt.xlim(0, 5000)

plt.subplot(1, 2, 2)
plt.plot(freqs[:len(dat_1)//4], np.abs(filtered_fft1[:len(dat_1)//4]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Filtered FFT - Channel 1')
plt.xlim(0, 5000)

plt.savefig('Filtered fft channel 1 signal.png')
plt.tight_layout()
plt.show()


# === FIGURE 2: Fourier amplitudes for channel 1 ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(freqs[:len(dat_1)//4], np.abs(fft_2[:len(dat_1)//4]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Original FFT - Channel 2')
plt.xlim(0, 5000)

plt.subplot(1, 2, 2)
plt.plot(freqs[:len(dat_1)//4], np.abs(filtered_fft2[:len(dat_1)//4]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Filtered FFT - Channel 2')
plt.xlim(0, 5000)

plt.savefig('Filtered fft channel 2 signal.png')
plt.tight_layout()
plt.show()


# === FIGURE 3: Time series for channel 0 ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(time[start_location:end_location], dat_1[start_location:end_location])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Time Series - Channel 1')

plt.subplot(1, 2, 2)
plt.plot(time[start_location:end_location], dat_1_filtered[start_location:end_location])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered Time Series - Channel 1')

plt.savefig('Filtered original channel 1 signal.png')
plt.tight_layout()
plt.show()


# === FIGURE 4: Time series for channel 1 ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(time[start_location:end_location], dat_2[start_location:end_location])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Time Series - Channel 2')

plt.subplot(1, 2, 2)
plt.plot(time[start_location:end_location], dat_2_filtered[start_location:end_location])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered Time Series - Channel 2')

plt.savefig('Filtered original channel 2 signal.png')
plt.tight_layout()
plt.show()


plt.tight_layout()
plt.show()

new_data = np.empty(data.shape, dtype=data.dtype)

new_data[:, 0] = dat_1_filtered.astype(data.dtype)
new_data[:, 1] = dat_2_filtered.astype(data.dtype)

write('GraviteaTime_changed.wav', sample_rate, new_data)