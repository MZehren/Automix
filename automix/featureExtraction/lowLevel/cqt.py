from typing import List

import librosa
import numpy as np

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class Cqt(Estimator):
    """
    Estimator calculating the cqt of given audio.

    parameter Scale possible values:
        Amplitude
        Power
        MSAF
        Power dB
        Perceived dB
    """

    def __init__(self,
                 parameterHopLength=512,
                 parameterBinNumber=84,
                 parameterScale="Power",
                 inputSamples="samples",
                 outputCqt="cqt",
                 cachingLevel=2,
                 forceRefreshCache=False):
        self.parameters = {
            "hopLength": Parameter(parameterHopLength),
            "binNumber": Parameter(parameterBinNumber),
            "scale": Parameter(parameterScale)
        }
        self.inputs = [inputSamples]
        self.outputs = [outputCqt]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, samples: Signal):
        """Calculates the cqt of the given audio using librosa.

        Args:
            samples (Signal): The samples of the audio.
            grid (list of float): The .

        Returns:
            tuple of List[float]: The cqt of the audio.

        """
        sr = samples.sampleRate
        hop_length = self.parameters["hopLength"].value
        n_bins = self.parameters["binNumber"].value
        cqt_sr = sr / hop_length
        cqt = librosa.cqt(samples.values, sr=sr, hop_length=hop_length, n_bins=n_bins)
        linear_cqt = np.abs(cqt)

        if self.parameters["scale"].value == "Amplitude":
            result = linear_cqt
        elif self.parameters["scale"].value == "Power":
            result = linear_cqt**2
        elif self.parameters["scale"].value == "MSAF":
            result = librosa.amplitude_to_db(linear_cqt**2, ref=np.max)
            result += np.min(result) * -1  # Inverting the db scale (don't know if this is correct)
        elif self.parameters["scale"].value == "Power dB":
            result = librosa.amplitude_to_db(linear_cqt, ref=np.max)  # Based on Librosa, standard power spectrum in dB
            result += np.min(result) * -1
        elif self.parameters["scale"].value == "Perceived dB":
            freqs = librosa.cqt_frequencies(linear_cqt.shape[0], fmin=librosa.note_to_hz('C1'))
            result = librosa.perceptual_weighting(linear_cqt**2, freqs, ref=np.max)
            result += np.min(result) * -1
        else:
            raise ValueError("parameterScale is not a correct value")

        return (Signal(result.T, sampleRate=cqt_sr), )
