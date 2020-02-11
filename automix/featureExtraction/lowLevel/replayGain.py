"""
container for the ReplayGain estimator
"""
from essentia import standard

from automix.featureExtraction.estimator import Estimator
from automix.model.classes.signal import Signal


# TODO replace by loudness?
class ReplayGain(Estimator):
    """
    Estimator computing the replayGain from the samples:
    I think 14dB of headroom
    """

    def __init__(self, inputSamples="samples", inputGrid="beats", output="replayGain", cachingLevel=0, forceRefreshCache=False):
        self.inputs = [inputSamples, inputGrid]
        self.outputs = [output]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache
        self.parameters = {}

    def predictOne(self, samples: Signal, grid: Signal):

        if grid is not None:
            values = [
                standard.ReplayGain(sampleRate=samples.sampleRate)(samples.getValues(grid.times[i], grid.times[i + 1]))
                for i in range(len(grid.times) - 1)
            ]
            return (Signal(values, times=grid.times[:-1]), )
        else:
            values = standard.ReplayGain(sampleRate=samples.sampleRate)(samples.values)
            return (Signal(values, times=[0]), )
        
        # except RuntimeError
        #   return ReplayGain(sampleRate=44100)(
        #     self.readAudioFile(path, sr=44100))
        #  see return (ld.loudnessSignal(samples, sampleRate), ) also