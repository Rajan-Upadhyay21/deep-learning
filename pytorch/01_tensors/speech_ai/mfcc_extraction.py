import numpy as np

sample_rate = 16000
time = np.linspace(0, 1, sample_rate, endpoint=False)
audio_signal = np.sin(2 * np.pi * 300 * time)

frame_size = 400
hop_length = 160
num_frames = 1 + (len(audio_signal) - frame_size) // hop_length

features = []
for i in range(num_frames):
    start = i * hop_length
    end = start + frame_size
    frame = audio_signal[start:end]
    energy = np.sum(frame ** 2)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    mean_val = np.mean(frame)
    features.append([energy, zero_crossings, mean_val])

features = np.array(features)

print("Pseudo-MFCC-style Feature Shape:", features.shape)
print("First 5 Feature Rows:")
print(features[:5])
