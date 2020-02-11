"""This module provides functions to calculate the perceptive loudness of
signals in different forms.

"""

from bisect import bisect_left
from essentia.standard import ReplayGain, Resample
from librosa.core import db_to_amplitude, istft


def loudnessSignal(y, sr):
    """Calculates the loudness of a signal.

    Args:
        y (list of float): The signal.
        sr (int, optional): The sample rate of the signal.

    Returns:
        float: The negative replay gain as the loudness in dB of the signal.

    """

    # Only these samplerates are accepted by ReplayGain()
    supportedSamplerates = [8000, 32000, 44100, 48000]

    if sr in supportedSamplerates:
        # Sample rate is okay. No need to change signal
        yNew = y
        srNew = sr
    else:
        # Samplerate is not okay
        # Resample the signal to fit

        # Find next higher supported sample rate
        idx = bisect_left(sorted(supportedSamplerates), sr)
        idx = min(idx, len(supportedSamplerates) - 1)
        srNew = supportedSamplerates[idx]
        # Resample signal
        fResample = Resample(
            inputSampleRate=sr, outputSampleRate=srNew, quality=0)
        yNew = fResample(y)

    fReplayGain = ReplayGain(sampleRate=srNew)
    loudness = -(fReplayGain(yNew) + 14)  # Offset replay gain by 14 dB
    return loudness


def normalizeSignal(y, sr):
    """Normalizes a signal by its loudness.

    Args:
        y (list of float): The signal.
        sr (int, optional): The sample rate of the signal.

    Returns:
        list of float: The signal normalized so that its loudness is 0.

    """

    loudness = loudnessSignal(y, sr)
    normalizationDivisor = db_to_amplitude(loudness)
    yNormalized = [sample / normalizationDivisor for sample in y]
    return yNormalized


def loudnessSTFTMatrix(matrix, sr, **kwargs):
    """Calculates the loudness of a signal encoded by its STFT matrix.

    Args:
        matrix (np.ndarray): STFT matrix of the actual signal.
        sr (int, optional): The sample rate of the input signal.
        **kwargs: Keywords for istft() (see
            https://librosa.github.io/librosa/generated/librosa.core.istft.html)

    Returns:
        float: The negative replay gain as the loudness in dB of the signal.

    """

    # Convert STFT matrix to signal and use loudnessSignal() to obtain loudness
    # TODO: this is inefficient
    y = istft(matrix, **kwargs)
    return loudnessSignal(y, sr)
