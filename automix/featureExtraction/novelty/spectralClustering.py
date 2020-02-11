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

    def predictOne(self, PCP: Signal, MFCC: Signal) -> Signal:
        """
        This script identifies the boundaries of a given track using the Spectral
        Clustering method published here:

        Mcfee, B., & Ellis, D. P. W. (2014). Analyzing Song Structure with Spectral
        Clustering. In Proc. of the 15th International Society for Music
        Information Retrieval Conference (pp. 405–410). Taipei, Taiwan.

        Original code by Brian McFee from:
            https://github.com/bmcfee/laplacian_segmentation
        """
        assert np.array_equal(PCP.times, MFCC.times)
        pass
