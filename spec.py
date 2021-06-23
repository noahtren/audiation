import numpy as np


def detrend_none(x, axis=None):
  """
    Return x: no detrending.
    Parameters
    ----------
    x : any object
        An object containing the data
    axis : int
        This parameter is ignored.
        It is included for compatibility with detrend_mean
    See Also
    --------
    detrend_mean : Another detrend algorithm.
    detrend_linear : Another detrend algorithm.
    detrend : A wrapper around all the detrend algorithms.
    """
  return x


def window_hanning(x):
  """
    Return x times the hanning window of len(x).
    See Also
    --------
    window_none : Another window algorithm.
    """
  return np.hanning(len(x)) * x


def stride_windows(x, n, noverlap=None, axis=0):
  """
    Get all windows of x with length n as a single array,
    using strides to avoid data duplication.
    .. warning::
        It is not safe to write to the output array.  Multiple
        elements may point to the same piece of memory,
        so modifying one value may change others.
    Parameters
    ----------
    x : 1D array or sequence
        Array or sequence containing the data.
    n : int
        The number of data points in each window.
    noverlap : int, default: 0 (no overlap)
        The overlap between adjacent windows.
    axis : int
        The axis along which the windows will run.
    References
    ----------
    `stackoverflow: Rolling window for 1D arrays in Numpy?
    <http://stackoverflow.com/a/6811241>`_
    `stackoverflow: Using strides for an efficient moving average filter
    <http://stackoverflow.com/a/4947453>`_
    """
  if noverlap is None:
    noverlap = 0

  if noverlap >= n:
    raise ValueError('noverlap must be less than n')
  if n < 1:
    raise ValueError('n cannot be less than 1')

  x = np.asarray(x)

  if x.ndim != 1:
    raise ValueError('only 1-dimensional arrays can be used')
  if n == 1 and noverlap == 0:
    if axis == 0:
      return x[np.newaxis]
    else:
      return x[np.newaxis].transpose()
  if n > x.size:
    raise ValueError('n cannot be greater than the length of x')

  # np.lib.stride_tricks.as_strided easily leads to memory corruption for
  # non integer shape and strides, i.e. noverlap or n. See #3845.
  noverlap = int(noverlap)
  n = int(n)

  step = n - noverlap
  if axis == 0:
    shape = (n, (x.shape[-1] - noverlap) // step)
    strides = (x.strides[0], step * x.strides[0])
  else:
    shape = ((x.shape[-1] - noverlap) // step, n)
    strides = (step * x.strides[0], x.strides[0])
  return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _spectral_helper(x,
                     y=None,
                     NFFT=None,
                     Fs=None,
                     detrend_func=None,
                     window=None,
                     noverlap=None,
                     pad_to=None,
                     sides=None,
                     scale_by_freq=None,
                     mode=None):
  """
    Private helper implementing the common parts between the psd, csd,
    spectrogram and complex, magnitude, angle, and phase spectrums.
    """
  if y is None:
    # if y is None use x for y
    same_data = True
  else:
    # The checks for if y is x are so that we can use the same function to
    # implement the core of psd(), csd(), and spectrogram() without doing
    # extra calculations.  We return the unaveraged Pxy, freqs, and t.
    same_data = y is x

  if Fs is None:
    Fs = 2
  if noverlap is None:
    noverlap = 0
  if detrend_func is None:
    detrend_func = detrend_none
  if window is None:
    window = window_hanning

  # if NFFT is set to None use the whole signal
  if NFFT is None:
    NFFT = 256

  if mode is None or mode == 'default':
    mode = 'psd'

  if not same_data and mode != 'psd':
    raise ValueError("x and y must be equal if mode is not 'psd'")

  # Make sure we're dealing with a numpy array. If y and x were the same
  # object to start with, keep them that way
  x = np.asarray(x)
  if not same_data:
    y = np.asarray(y)

  if sides is None or sides == 'default':
    if np.iscomplexobj(x):
      sides = 'twosided'
    else:
      sides = 'onesided'

  # zero pad x and y up to NFFT if they are shorter than NFFT
  if len(x) < NFFT:
    n = len(x)
    x = np.resize(x, NFFT)
    x[n:] = 0

  if not same_data and len(y) < NFFT:
    n = len(y)
    y = np.resize(y, NFFT)
    y[n:] = 0

  if pad_to is None:
    pad_to = NFFT

  if mode != 'psd':
    scale_by_freq = False
  elif scale_by_freq is None:
    scale_by_freq = True

  # For real x, ignore the negative frequencies unless told otherwise
  if sides == 'twosided':
    numFreqs = pad_to
    if pad_to % 2:
      freqcenter = (pad_to - 1) // 2 + 1
    else:
      freqcenter = pad_to // 2
    scaling_factor = 1.
  elif sides == 'onesided':
    if pad_to % 2:
      numFreqs = (pad_to + 1) // 2
    else:
      numFreqs = pad_to // 2 + 1
    scaling_factor = 2.

  if not np.iterable(window):
    window = window(np.ones(NFFT, x.dtype))
  if len(window) != NFFT:
    raise ValueError("The window length must match the data's first dimension")

  result = stride_windows(x, NFFT, noverlap, axis=0)
  # result = detrend(result, detrend_func, axis=0)
  result = result * window.reshape((-1, 1))
  result = np.fft.fft(result, n=pad_to, axis=0)[:numFreqs, :]
  freqs = np.fft.fftfreq(pad_to, 1 / Fs)[:numFreqs]

  if not same_data:
    # if same_data is False, mode must be 'psd'
    resultY = stride_windows(y, NFFT, noverlap)
    # resultY = detrend(resultY, detrend_func, axis=0)
    resultY = resultY * window.reshape((-1, 1))
    resultY = np.fft.fft(resultY, n=pad_to, axis=0)[:numFreqs, :]
    result = np.conj(result) * resultY
  elif mode == 'psd':
    result = np.conj(result) * result
  elif mode == 'magnitude':
    result = np.abs(result) / np.abs(window).sum()
  elif mode == 'angle' or mode == 'phase':
    # we unwrap the phase later to handle the onesided vs. twosided case
    result = np.angle(result)
  elif mode == 'complex':
    result /= np.abs(window).sum()

  if mode == 'psd':

    # Also include scaling factors for one-sided densities and dividing by
    # the sampling frequency, if desired. Scale everything, except the DC
    # component and the NFFT/2 component:

    # if we have a even number of frequencies, don't scale NFFT/2
    if not NFFT % 2:
      slc = slice(1, -1, None)
    # if we have an odd number, just don't scale DC
    else:
      slc = slice(1, None, None)

    result[slc] *= scaling_factor

    # MATLAB divides by the sampling frequency so that density function
    # has units of dB/Hz and can be integrated by the plotted frequency
    # values. Perform the same scaling here.
    if scale_by_freq:
      result /= Fs
      # Scale the spectrum by the norm of the window to compensate for
      # windowing loss; see Bendat & Piersol Sec 11.5.2.
      result /= (np.abs(window)**2).sum()
    else:
      # In this case, preserve power in the segment, not amplitude
      result /= np.abs(window).sum()**2

  t = np.arange(NFFT / 2, len(x) - NFFT / 2 + 1, NFFT - noverlap) / Fs

  if sides == 'twosided':
    # center the frequency range at zero
    freqs = np.roll(freqs, -freqcenter, axis=0)
    result = np.roll(result, -freqcenter, axis=0)
  elif not pad_to % 2:
    # get the last value correctly, it is negative otherwise
    freqs[-1] *= -1

  # we unwrap the phase here to handle the onesided vs. twosided case
  if mode == 'phase':
    result = np.unwrap(result, axis=0)

  return result, freqs, t
