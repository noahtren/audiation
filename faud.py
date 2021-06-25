import os
from io import BytesIO

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC
import streamlit as st
import matplotlib.pyplot as plt
import soundfile as sf
from matplotlib import cm
from tqdm import tqdm

from spec import _spectral_helper
from aug import augment_signal


def torch_to_py(tensor):
  return tensor.detach().cpu().numpy()


@st.cache(hash_funcs={torch.nn.parameter.Parameter: id})
def get_model():
  model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base").cuda()
  return model


def freq_coeff_power(coeff):
  freqs = torch.fft.fftfreq(coeff.shape[-1],
                            d=1 / coeff.shape[-1]).unsqueeze(0).abs().cuda()
  return coeff * (freqs / coeff.shape[-1])


def normalize_freq_coeff(coeff):
  freqs = torch.fft.fftfreq(coeff.shape[-1]).unsqueeze(0).abs().cuda()
  scale = (1 / (freqs + 1))
  return coeff * scale


def plot_tensor_dict(tdict, title=None):
  fig, ax = plt.subplots()
  _ = [
      ax.plot(torch_to_py(torch.Tensor(v)), label=k) for k, v in tdict.items()
  ]
  ax.legend()
  if title:
    ax.set_title(title)
  st.pyplot(fig)


def display_spectrogram(signal, audio_widget=False, do_plot=True):
  assert signal.ndim == 1
  fig, axes = plt.subplots(nrows=2,
                           figsize=[10, 6],
                           gridspec_kw={'height_ratios': [2, 1]})
  result, _, _ = _spectral_helper(signal, NFFT=256, noverlap=128 + 64 + 32)
  Z = np.log(result.real)
  spec_data = axes[0].imshow(np.flipud(Z),
                             cmap='plasma',
                             aspect='auto',
                             vmin=-20,
                             vmax=0)
  # fig.colorbar(spec_data, ax=axes[0])
  axes[0].axis('off')
  axes[1].plot(signal, linewidth=0.7, color='black')
  axes[1].axis('off')
  axes[1].set_ylim(-0.03, 0.03)
  if do_plot:
    st.markdown(
        f"{signal.shape[0] / 16} ms, {signal.shape}, {signal.max()}, {signal.min()}"
    )
    st.pyplot(fig)
  if audio_widget:
    bytes_io = BytesIO(bytes())
    sf.write(bytes_io, signal, 16000, format="wav")
    st.audio(bytes_io)
  return fig


def token_optimize(signal_length: int,
                   optim_steps: int,
                   lr: float,
                   layer_nums: int,
                   neuron_nums: int,
                   activity_reg=False,
                   target_middle=True,
                   log_func=None,
                   plot_metrics=False,
                   save_snapshots=False,
                   model=None):
  state = {}
  assert model is not None, "Model must be passed to this function"

  def layer_index_update(i):
    def update_values(self, input, output):
      state[i] = output

    return update_values

  for layer_num in range(12):
    model.wav2vec2.encoder.layers[
        layer_num].final_layer_norm.register_forward_hook(
            layer_index_update(layer_num))

  params = {
      'theta':
      torch.normal(0,
                   0.0001,
                   size=[len(layer_nums), signal_length],
                   dtype=torch.cfloat).cuda().requires_grad_(),
  }
  optim = torch.optim.Adam(params.values(), lr=lr)
  metrics = {
      'avg_activation_l1': [],
      'avg_activation_l2': [],
      'average_power': [],
      'average_residual': [],
  }
  losses = {
      'avg_activation_l1': [],
      'target_activation': [],
  }
  for i in tqdm(range(optim_steps)):
    # optimize step
    optim.zero_grad()
    signal = torch.fft.ifft(normalize_freq_coeff(params['theta'])).real
    average_power = (signal**2).mean()
    model(augment_signal(signal))
    stacked_state = torch.stack([state[i] for i in range(12)])
    if target_middle:
      mid = stacked_state.shape[2] // 2
      target_activations = stacked_state[layer_nums,
                                         np.arange(len(layer_nums)), mid,
                                         neuron_nums]
    else:
      target_activations = stacked_state[layer_nums,
                                         np.arange(len(layer_nums)), :,
                                         neuron_nums]
    layer_numel = stacked_state.shape[-2:].numel()
    metrics['avg_activation_l1'].append(
        (stacked_state[layer_nums, np.arange(len(layer_nums))].abs().sum() -
         target_activations.abs().sum()) / (layer_numel))
    metrics['avg_activation_l2'].append(
        (stacked_state[layer_nums, np.arange(len(layer_nums))].pow(2).sum() -
         target_activations.pow(2).sum()) / (layer_numel))
    metrics['average_power'].append(average_power)
    metrics['average_residual'].append(signal.abs().mean())
    losses['avg_activation_l1'].append(metrics['avg_activation_l1'][-1] * 1e-5)
    losses['target_activation'].append(target_activations.mean() * 2e-5)
    loss = sum([v[-1] for v in losses.values()])
    loss.backward()
    optim.step()
    for m in metrics.values():
      m[-1] = m[-1].detach()
    for l in losses.values():
      l[-1] = l[-1].detach()
    if log_func:
      for k, v in losses.items():
        log_func('loss.' + k, torch_to_py(v[-1]).tolist(), i)
      for k, v in metrics.items():
        log_func('metric.' + k, torch_to_py(v[-1]).tolist(), i)
    if save_snapshots and i % 16 == 0:
      os.makedirs("./imgs", exist_ok=True)
      fig = display_spectrogram(torch_to_py(signal[0]), do_plot=False)
      fig.savefig(f'./imgs/{i // 16}.png')
    # regularize steps
    for _ in range(4):
      optim.zero_grad()
      loss = freq_coeff_power(params['theta']).abs().pow(
          2).mean() * 0.006 + params['theta'].abs().pow(2).mean() * 2e-3
      loss.backward()
      optim.step()

  signal = torch.fft.ifft(normalize_freq_coeff(params['theta'])).real
  if plot_metrics:
    plot_tensor_dict(losses, title="losses")
    plot_tensor_dict(metrics, title="metrics")
  return torch_to_py(signal)


if __name__ == "__main__":
  model = get_model()
  layer_nums = np.repeat(np.array([0, 1]), 1)
  neuron_nums = np.tile(np.array([0]), 1)
  col1, col2 = st.beta_columns(2)
  with col1:
    signals = token_optimize(16384,
                             128,
                             0.001,
                             layer_nums,
                             neuron_nums,
                             target_middle=True,
                             save_snapshots=False,
                             plot_metrics=True,
                             model=model)
  with col2:
    fig = display_spectrogram(signals[0] * 20, audio_widget=True)
    fig = display_spectrogram(signals[1] * 20, audio_widget=True)
