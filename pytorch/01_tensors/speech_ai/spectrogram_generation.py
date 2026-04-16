import numpy as np
import matplotlib.pyplot as plt

sample_rate = 16000
time = np.linspace(0, 2, sample_rate * 2, endpoint=False)
audio_signal = 0.5 * np.sin(2 * np.pi * 220 * time) + 0.3 * np.sin(2 * np.pi * 440 * time)

plt.figure(figsize=(10, 4))
plt.specgram(audio_signal, Fs=sample_rate, NFFT=256, noverlap=128)
plt.title("Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(label="Intensity")
plt.show()
