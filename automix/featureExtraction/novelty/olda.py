from msaf.algorithms.olda import segmenter

import numpy as np
from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class OLDA(Estimator):
    """
    Estimator computing the olda novelty from McFee 2014
    Implementation from msaf
    """

    def __init__(self, inputSamples="normalizedBarMSE", outputNovelty="noveltyMSE", cachingLevel=2, forceRefreshCache=False):
        self.parameters = {}
        self.inputs = [inputSamples]
        self.outputs = [outputNovelty]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, samples: Signal) -> Signal:
        """
        TODO
        """
        W = np.load("/home/mickael/Documents/programming/article-msa-structure/msaf/msaf/algorithms/olda/models/EstBeats_BeatlesTUT.npy")
        
        F = W.dot(samples.values)

        kmin, kmax = segmenter.get_num_segs(samples.duration)
        est_idxs = segmenter.get_segments(F, kmin=kmin, kmax=kmax)

        return (est_idxs, )
