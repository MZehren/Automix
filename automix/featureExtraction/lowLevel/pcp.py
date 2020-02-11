from typing import List

import librosa
import numpy as np

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class Pcp(Estimator):
    """
    Estimator calculating the cqt of given audio.

    parameter Scale possible values:
        Amplitude
        Power
        MSAF
        Power dB
        Perceived dB
    """

    def __init__(
            self,
            parameterHopLength=512,
            parameterNieto=False,  # Use nieto's implementation
            #  parameterScale="Power",
            inputSamples="samples",
            outputPcp="pcp",
            cachingLevel=2,
            forceRefreshCache=False):
        self.parameters = {"hopLength": Parameter(parameterHopLength), "nieto": Parameter(parameterNieto)}
        self.inputs = [inputSamples]
        self.outputs = [outputPcp]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, samples: Signal):
        """Calculates the pcp of the given audio using linrosa.feature.chroma_stft

        Args:
            samples (Signal): The samples of the audio.

        Returns:
            tuple (1,Signal(samples, bins)): The pcp of the audio.
        """
        if self.parameters["nieto"].value:
            return self.nietoPCP(samples)
        else:
            return self.chromagram(samples)

    def nietoPCP(self, samples: Signal):
        sr = samples.sampleRate
        hop_length = self.parameters["hopLength"].value
        pcp_sr = sr / hop_length

        audio_harmonic, _ = librosa.effects.hpss(samples.values)
        # I double checked, and the parameters are the one used in MSAF. 7 octave in pcp_cqt and 6 octaves in pcp
        pcp_cqt = np.abs(librosa.hybrid_cqt(audio_harmonic, sr=sr, hop_length=hop_length, n_bins=7 * 12, norm=np.inf,
                                            fmin=27.5))**2
        pcp = librosa.feature.chroma_cqt(C=pcp_cqt, sr=sr, hop_length=hop_length, n_octaves=6, fmin=27.5).T

        return (Signal(pcp, sampleRate=pcp_sr), )

    def chromagram(self, samples: Signal):
        sr = samples.sampleRate
        result = librosa.feature.chroma_stft(y=samples.values, sr=sr)
        hop_length = self.parameters["hopLength"].value
        pcp_sr = sr / hop_length

        return (Signal(result.T, sampleRate=pcp_sr), )
