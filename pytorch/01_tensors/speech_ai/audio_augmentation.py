import numpy as np

sample_rate = 16000
time = np.linspace(0, 1, sample_rate, endpoint=False)
audio_signal = np.sin(2 * np.pi * 440 * time)

noise = 0.02 * np.random.randn(len(audio_signal))
noisy_signal = audio_signal + noise

shift = 500
shifted_signal = np.roll(audio_signal, shift)

volume_scaled_signal = 1.2 * audio_signal

print("Original Signal Shape:", audio_signal.shape)
print("Noisy Signal First 5 Samples:", noisy_signal[:5])
print("Shifted Signal First 5 Samples:", shifted_signal[:5])
print("Volume Scaled Signal First 5 Samples:", volume_scaled_signal[:5])
