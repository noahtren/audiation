import torch
import julius
import numpy as np


def augment_signal(signal,
                   aug_settings={'resample', 'shift', 'additive_noise'},
                   device='cuda'):
  if 'resample' in aug_settings:
    orig_size = signal.shape[-1]
    new_freq = 160 + int(((np.random.uniform() * 2) - 1) * 16)
    signal = julius.resample_frac(signal, 160, new_freq)
    size_diff = abs(signal.shape[-1] - orig_size)
    if signal.shape[-1] > orig_size:
      signal = signal[..., size_diff // 2:-(size_diff // 2 - size_diff % 2)]
    else:
      signal = torch.nn.functional.pad(
          signal, [size_diff // 2, size_diff // 2 + size_diff % 2])
  if 'additive_noise' in aug_settings:
    signal = signal + torch.normal(0, 0.0007, size=signal.shape, device=device)
  if 'random_scale' in aug_settings:
    signal = signal * torch.normal(1, 0.05, size=[], device=device)
  if 'shift' in aug_settings:
    shift_range = 16
    signal = torch.nn.functional.pad(signal, [shift_range, shift_range])
    shift_amount = ((torch.rand([]) * 2 - 1) * shift_range).int()
    signal = signal[:, shift_range - shift_amount:-shift_range - shift_amount]
  return signal
