"""
container for the ReadFile estimator
"""
import librosa

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class ReadFile(Estimator):
    """
    estimator reading a file from a path
    TODO: the output here should not be serialized !
    """

    def __init__(self,
                 inputPath="path",
                 outputSamples="samples",
                 parameterSampleRate=None,
                 cachingLevel=2,
                 forceRefreshCache=False):
        self.inputs = [inputPath]
        self.outputs = [outputSamples]
        self.parameters = {"sampleRate": Parameter(parameterSampleRate)}
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, path):
        y, sr = librosa.load(path, sr=self.parameters["sampleRate"].value)
        return (Signal(y, sampleRate=sr), )


class GetDuration(Estimator):
    """
    Work around to serialize the exact duration of the tracks
    """
    def __init__(self, inputs=["samples"], outputs=["duration"]):
        super().__init__(inputs=inputs, outputs=outputs)

    def predictOne(self, samples):
        return (samples.duration,)