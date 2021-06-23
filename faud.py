from io import BytesIO, FileIO

import torch
import librosa
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
def get_models():
  model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base").cuda()
  return model


def freq_coeff_power(coeff):
  freqs = torch.fft.fftfreq(coeff.shape[-1],
                            d=1 / coeff.shape[-1]).unsqueeze(0).abs().cuda()
  return coeff * (freqs / coeff.shape[-1])


def normalize_freq_coeff(coeff):
  freqs = torch.fft.fftfreq(coeff.shape[-1],
                            d=1 / coeff.shape[-1]).unsqueeze(0).abs().cuda()
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
  fig, axes = plt.subplots(nrows=2, figsize=[10, 8])
  result, _, _ = _spectral_helper(signal, NFFT=256, noverlap=128)
  Z = np.log(result.real)
  spec_data = axes[0].imshow(np.flipud(Z),
                             cmap='winter',
                             aspect='auto',
                             vmin=-20,
                             vmax=0)
  # fig.colorbar(spec_data, ax=axes[0])
  axes[0].axis('off')
  axes[1].scatter(np.arange(signal.shape[0]), signal, c=cm.jet(np.abs(signal)))
  axes[1].axis('off')
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


def contextual_token_optimize_many(signal_length: int,
                                   optim_steps: int,
                                   lr: float,
                                   layer_nums: int,
                                   neuron_nums: int,
                                   activity_reg: bool,
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

  # initialize
  params = {
      'theta':
      torch.normal(0,
                   1,
                   size=[len(layer_nums), signal_length],
                   dtype=torch.cfloat).cuda().requires_grad_(),
  }
  optim = torch.optim.Adam(params.values(), lr=lr)
  metrics = {
      'avg_activation_l1': [],
      'target_activation': [],
      'average_power': [],
      'average_residual': []
  }
  losses = {
      'avg_activation_l1': [],
      'target_activation': [],
  }
  if activity_reg:
    losses['average_power'] = []
    losses['average_residual'] = []
  for i in tqdm(range(optim_steps)):
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
      target_activations = stacked_state[layer_nums, np.arange(len(layer_nums)), :, neuron_nums]
    metrics['avg_activation_l1'].append(stacked_state[layer_nums].abs().mean())
    metrics['target_activation'].append(target_activations.mean())
    metrics['average_power'].append(average_power)
    metrics['average_residual'].append(signal.abs().mean())
    losses['avg_activation_l1'].append(stacked_state[layer_nums].abs().mean())
    losses['target_activation'].append(target_activations.mean() * -32)
    if activity_reg:
      losses['average_power'].append(average_power * 1024)
      losses['average_residual'].append(signal.abs().mean() * 32)
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
      fig = display_spectrogram(torch_to_py(signal[0]), do_plot=False)
      fig.savefig(f'./imgs/{i // 16}.png')

  signal = torch.fft.ifft(normalize_freq_coeff(params['theta'])).real
  if plot_metrics:
    plot_tensor_dict(losses, title="losses")
    plot_tensor_dict(metrics, title="metrics")
  return torch_to_py(signal)


def caricature_optimize(orig_signal,
                        layer_nums,
                        optim_steps: int,
                        lr: float,
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

  model(orig_signal)
  orig_state = torch.stack([state[i] for i in range(12)]).detach()
  params = {
      'theta':
      torch.normal(0,
                   1,
                   size=[len(layer_nums), orig_signal.shape[-1]],
                   dtype=torch.cfloat).cuda().requires_grad_(),
  }
  optim = torch.optim.Adam(params.values(), lr=lr)
  metrics = {
      'target_activation': [],
      'average_power': [],
      'average_residual': []
  }
  losses = {
      'target_activation': [],
      'average_power': []
  }
  for i in tqdm(range(optim_steps)):
    optim.zero_grad()
    signal = torch.fft.ifft(normalize_freq_coeff(params['theta'])).real
    signal = torch.tanh(signal)
    average_power = (signal**2).mean()
    model(augment_signal(signal))
    stacked_state = torch.stack([state[i] for i in range(12)])
    target_activations = (stacked_state[layer_nums] -
                          orig_state[layer_nums])**2
    metrics['target_activation'].append(target_activations.mean())
    metrics['average_power'].append(average_power)
    metrics['average_residual'].append(signal.abs().mean())
    losses['target_activation'].append(target_activations.mean())
    losses['average_power'].append(average_power)
    loss = sum([v[-1] for v in losses.values()])
    loss.backward()
    optim.step()
  signal = torch.fft.ifft(normalize_freq_coeff(params['theta'])).real
  plot_tensor_dict(losses, title="losses")
  plot_tensor_dict(metrics, title="metrics")
  return torch_to_py(torch.tanh(signal))


if __name__ == "__main__":
  CARICATURE = False
  CONTEXTUAL_NEURON = True

  # get example audio and model
  example_aud, sr = librosa.load('wait a minute.mp3')
  example_aud = librosa.resample(example_aud, sr, 16000)
  if example_aud.ndim == 2:
    example_aud = example_aud[:, 0]
  model = get_models()

  if CARICATURE:
    layer_nums = np.array([2])
    signals = caricature_optimize(torch.Tensor(example_aud).unsqueeze(0).cuda(),
                                  layer_nums,
                                  2000,
                                  1024,
                                  model=model)
    display_spectrogram(signals[0], audio_widget=True)

  if CONTEXTUAL_NEURON:
    layer_nums = np.repeat(np.arange(4), 2)
    neuron_nums = np.tile(np.array([0, 1]), 4)
    signals = contextual_token_optimize_many(8000,
                                             4096,
                                             16,
                                             layer_nums,
                                             neuron_nums,
                                             True,
                                             target_middle=True,
                                             save_snapshots=True,
                                             model=model)
    col1, col2 = st.beta_columns(2)
    with col1:
      fig = display_spectrogram(signals[0], audio_widget=True)
      fig = display_spectrogram(signals[1], audio_widget=True)
      fig = display_spectrogram(signals[2], audio_widget=True)
      fig = display_spectrogram(signals[3], audio_widget=True)
    with col2:
      fig = display_spectrogram(signals[4], audio_widget=True)
      fig = display_spectrogram(signals[5], audio_widget=True)
      fig = display_spectrogram(signals[6], audio_widget=True)
      fig = display_spectrogram(signals[7], audio_widget=True)
