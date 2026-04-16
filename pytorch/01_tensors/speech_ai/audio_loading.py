
Code files:

## `audio_loading.py`

```python
import numpy as np

sample_rate = 16000
duration_seconds = 1
time = np.linspace(0, duration_seconds, sample_rate * duration_seconds, endpoint=False)

audio_signal = 0.5 * np.sin(2 * np.pi * 440 * time)

print("Sample Rate:", sample_rate)
print("Duration (seconds):", duration_seconds)
print("Audio Signal Shape:", audio_signal.shape)
print("First 10 Samples:")
print(audio_signal[:10])
