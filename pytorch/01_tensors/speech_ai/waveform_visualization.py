import numpy as np
import matplotlib.pyplot as plt

sample_rate = 16000
time = np.linspace(0, 1, sample_rate, endpoint=False)
audio_signal = 0.5 * np.sin(2 * np.pi * 220 * time)

plt.figure(figsize=(10, 4))
plt.plot(time[:1000], audio_signal[:1000])
plt.title("Waveform Visualization")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
