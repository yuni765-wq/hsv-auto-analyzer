# -*- coding: utf-8 -*-
import imageio, numpy as np
from hooks import simulate_sequence


if __name__ == '__main__':
frames = simulate_sequence(seconds=2.0, fps=30, amp=2.5, freq_hz=4.0)
frames_u8 = (np.clip(frames, 0, 1) * 255).astype('uint8')
imageio.mimsave('artifacts/sim/preview.gif', frames_u8, duration=1/30)
print('Saved artifacts/sim/preview.gif')