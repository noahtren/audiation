import torch
import julius
import numpy as np
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm


def augment_signal(
    signal,
    aug_settings={'resample', 'additive_noise', 'random_scale', 'shift'},
    device='cuda'):
  if 'resample' in aug_settings:
    orig_size = signal.shape[-1]
    new_freq = 160 + int(((np.random.uniform() * 2) - 1) * 20)
    signal = julius.resample_frac(signal, 160, new_freq)
    size_diff = abs(signal.shape[-1] - orig_size)
    if signal.shape[-1] > orig_size:
      signal = signal[..., size_diff // 2:-(size_diff // 2 - size_diff % 2)]
    else:
      signal = torch.nn.functional.pad(
          signal, [size_diff // 2, size_diff // 2 + size_diff % 2])
  if 'additive_noise' in aug_settings:
    signal = signal + torch.normal(0, 0.05, size=signal.shape, device=device)
  # random scale
  if 'random_scale' in aug_settings:
    signal = signal * torch.normal(1, 0.05, size=[], device=device)
  if 'shift' in aug_settings:
    shift_range = 256
    signal = torch.nn.functional.pad(signal, [shift_range, shift_range])
    shift_amount = ((torch.rand([]) * 2 - 1) * shift_range).int()
    signal = signal[:, shift_range - shift_amount:-shift_range - shift_amount]
  return signal


if __name__ == "__main__":
  import streamlit as st
  from faud import display_spectrogram
  st.markdown("## Audio augmentations")
  example_aud, sr = librosa.load('wait a minute.mp3')
  example_aud = librosa.resample(example_aud, sr, 16000)
  col1, col2 = st.beta_columns(2)
  with col1:
    display_spectrogram(example_aud)
  with col2:
    for _ in tqdm(range(3)):
      aug = augment_signal(torch.Tensor(example_aud[np.newaxis]).cuda(),
                           device='cuda').cpu().numpy()[0]
      display_spectrogram(aug)
