from typing import List

import librosa
import numpy as np

from automix.featureExtraction.estimator import Estimator
from automix.model.classes.signal import Signal


class Hpss(Estimator):
    """
    Estimator computing the hpss of given audio.
    """

    def __init__(self,
                 inputSamples="samples",
                 outputHarmonic="harmonic",
                 outputPercussive="percussive",
                 cachingLevel=2,
                 forceRefreshCache=False):
        super().__init__()
        self.inputs = [inputSamples]
        self.outputs = [outputHarmonic, outputPercussive]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, samples: Signal) -> List[Signal]:
        """
        Computes the hpss of the given audio using librosa.
        """

        y_harmonic, y_percussive = librosa.effects.hpss(samples.values)

        return (Signal(np.array(y_harmonic), sampleRate=samples.sampleRate),
                Signal(np.array(y_percussive), sampleRate=samples.sampleRate))
