import numpy as np
import scipy.signal
from typing import Tuple, Dict

def detect_bad_channels(
    raw: np.ndarray, 
    fs: float, 
    similarity_threshold: Tuple[float, float] = (-0.5, 1), 
    psd_hf_threshold: float = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Detects bad channels in Neuropixel probe recordings.

    Channels are labeled as follows:
    - 0: Clear
    - 1: Dead (low coherence/amplitude)
    - 2: Noisy
    - 3: Outside of the brain

    Parameters:
    - raw: (nc, ns) numpy array of raw LFP data with nc channels and ns samples.
    - fs: Sampling frequency in Hz.
    - similarity_threshold: Tuple specifying the cross-correlation thresholds to identify bad channels.
    - psd_hf_threshold: Threshold for the high-frequency power spectral density to flag noisy channels.

    Returns:
    - labels: A numpy array of size (nc,) with the labels for each channel.
    - xfeats: A dictionary containing computed features for each channel.
    """
    
    def _detrend(x: np.ndarray, nmed: int) -> np.ndarray:
        """Subtracts the trend using median filtering."""
        ntap = int(np.ceil(nmed / 2))
        xf = np.r_[np.zeros(ntap) + x[0], x, np.zeros(ntap) + x[-1]]
        xf = scipy.signal.medfilt(xf, nmed)[ntap:-ntap]
        return x - xf

    def _channels_similarity(raw: np.ndarray, nmed: int = 0) -> np.ndarray:
        """Computes zero-lag cross-correlation similarity of each channel with the median trace."""
        def fxcor(x, y):
            return scipy.fft.irfft(scipy.fft.rfft(x) * np.conj(scipy.fft.rfft(y)), n=raw.shape[-1])

        def nxcor(x, ref):
            ref = ref - np.mean(ref)
            apeak = fxcor(ref, ref)[0]
            x = x - np.mean(x, axis=-1)[:, np.newaxis]
            return fxcor(x, ref)[:, 0] / apeak

        ref = np.median(raw, axis=0)
        xcor = nxcor(raw, ref)

        if nmed > 0:
            xcor = _detrend(xcor, nmed) + 1
        return xcor

    nc, _ = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # Remove DC offset
    xcor = _channels_similarity(raw)
    fscale, psd = scipy.signal.welch(raw * 1e6, fs=fs)  # Compute power spectral density
    
    if psd_hf_threshold is None:
        psd_hf_threshold = 1.4 if fs < 5000 else 0.02

    sos_hp = scipy.signal.butter(N=3, Wn=300 / fs * 2, btype='highpass', output='sos')
    hf = scipy.signal.sosfiltfilt(sos_hp, raw)
    xcorf = _channels_similarity(hf)

    xfeats = {
        'ind': np.arange(nc),
        'rms_raw': np.sqrt(np.mean(raw**2, axis=-1)),  # Root mean square of raw data
        'xcor_hf': _detrend(xcor, 11),
        'xcor_lf': xcorf - _detrend(xcorf, 11) - 1,
        'psd_hf': np.mean(psd[:, fscale > (fs / 2 * 0.8)], axis=-1),  # High-frequency power
    }

    # Channel classification
    ichannels = np.zeros(nc)
    idead = np.where(xfeats['xcor_hf'] < similarity_threshold[0])[0]
    inoisy = np.where((xfeats['psd_hf'] > psd_hf_threshold) | (xfeats['xcor_hf'] > similarity_threshold[1]))[0]
    ioutside = np.where(xfeats['xcor_lf'] < -0.75)[0]

    if ioutside.size > 0 and ioutside[-1] == (nc - 1):
        ioutside = ioutside[np.cumsum(np.r_[0, np.diff(ioutside) - 1]) == np.max(np.cumsum(np.r_[0, np.diff(ioutside) - 1]))]
        ichannels[ioutside] = 3

    ichannels[idead] = 1
    ichannels[inoisy] = 2

    return ichannels, xfeats
