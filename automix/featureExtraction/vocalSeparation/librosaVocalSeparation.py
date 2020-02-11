"""
container for the vocal separation estimator
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class LibrosaVocalSeparation(Estimator):
    """
    estimator based on http://librosa.github.io/librosa/auto_examples/plot_vocal_separation.html?highlight=vocalness
    """

    def __init__(self,
                 inputPath="samples",
                 outputVocals="vocals",
                 outoutBackground="background",
                 cachingLevel=2,
                 forceRefreshCache=False):
        self.inputs = [inputPath]
        self.outputs = [outputVocals, outoutBackground]
        self.parameters = {}
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, samples: Signal):
        """
        """
        y, sr = samples.values, samples.sampleRate

        # And compute the spectrogram magnitude and phase
        S_full, phase = librosa.magphase(librosa.stft(y))

        hopLength = 2048 / 4
        newSampleRate = sr / hopLength
        
        # We'll compare frames using cosine similarity, and aggregate similar frames
        # by taking their (per-frequency) median value.
        #
        # To avoid being biased by local continuity, we constrain similar frames to be
        # separated by at least 2 seconds.
        #
        # This suppresses sparse/non-repetetitive deviations from the average spectrum,
        # and works well to discard vocal elements.
        S_filter = librosa.decompose.nn_filter(S_full,
                                               aggregate=np.median,
                                               metric='cosine',
                                               width=int(librosa.time_to_frames(2, sr=sr)))

        # The output of the filter shouldn't be greater than the input
        # if we assume signals are additive.  Taking the pointwise minimium
        # with the input spectrum forces this.
        S_filter = np.minimum(S_full, S_filter)

        # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
        # Note: the margins need not be equal for foreground and background separation
        margin_i, margin_v = 2, 10
        power = 2

        mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)

        mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)

        # Once we have the masks, simply multiply them with the input spectrum
        # to separate the components
        S_foreground = mask_v * S_full
        S_background = mask_i * S_full

        # # sphinx_gallery_thumbnail_number = 2
        # idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
        # plt.figure(figsize=(12, 8))
        # plt.subplot(3, 1, 1)
        # librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
        #                         y_axis='log', sr=sr)
        # plt.title('Full spectrum')
        # plt.colorbar()

        # plt.subplot(3, 1, 2)
        # librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
        #                         y_axis='log', sr=sr)
        # plt.title('Background')
        # plt.colorbar()
        # plt.subplot(3, 1, 3)
        # librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
        #                         y_axis='log', x_axis='time', sr=sr)
        # plt.title('Foreground')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.show()

        return (Signal(S_foreground.T, sampleRate=newSampleRate), Signal(S_background.T, sampleRate=newSampleRate))
