import copy
from typing import List

import librosa
import numpy as np
from scipy.spatial import distance
from scipy import signal

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class Checkerboard(Estimator):
    """
    Estimator computing the checkerboard novelty from Foote
    Implementation from msaf

    Parameters:
    - parameterWindowSize = size of the checkerboard kernel for the convolution in ticks of the input
      The kernel is split on the middle, so use twice the wanted size
    - parameterDistanceMetric = how to compute the distance between samples to create the ssm. 
      Veire uses cosine distance for tensor, absolute scalar difference for a scalar
    - parameterDebugViz: deprecated
    - addZerosStart: pad the start of the file with either 0, or the first sample to be able to apply the convolution right from the start of the file
      accepts values [None, True, 0]
    """

    def __init__(self,
                 parameterWindowSize=16,
                 parameterDistanceMetric="seuclidean",
                 parameterDebugViz=False,
                 parameterAddZerosStart=True,
                 inputSamples="normalizedBarMSE",
                 outputNovelty="noveltyMSE",
                 cachingLevel=2,
                 forceRefreshCache=False):
        self.parameters = {
            "windowSize": Parameter(parameterWindowSize),
            "distanceMetric": Parameter(parameterDistanceMetric),
            "debugViz": Parameter(parameterDebugViz),
            "addZerosStart": Parameter(parameterAddZerosStart)
        }
        self.inputs = [inputSamples]
        self.outputs = [outputNovelty]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def compute_gaussian_krnl(self, M):
        """Creates a gaussian kernel following Foote's paper."""
        g = signal.gaussian(M, M // 3., sym=True)
        G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
        G[M // 2:, :M // 2] = -G[M // 2:, :M // 2]
        G[:M // 2, M // 2:] = -G[:M // 2, M // 2:]
        return G

    def compute_ssm(self, X, metric="seuclidean"):
        """Computes the self-similarity matrix of X."""
        D = distance.pdist(X, metric=metric)
        D = distance.squareform(D)
        D /= D.max()  # TODO: Why normalizing here ?
        return 1 - D

    def compute_nc(self, X, G):
        """Computes the novelty curve from the self-similarity matrix X and
            the gaussian kernel G."""
        N = X.shape[0]
        M = G.shape[0]
        nc = np.zeros(N)

        # Convolution on the diagonal
        for i in range(M // 2, N - M // 2 + 1):
            nc[i] = np.sum(X[i - M // 2:i + M // 2, i - M // 2:i + M // 2] * G)

        # Normalize
        # TODO: Why normalizing here ??
        nc += nc.min()
        nc /= nc.max()
        return nc

    def predictOne(self, samples: Signal) -> Signal:
        """
        see Foot 2000
        """
        # Make the input array multidimensional
        f = samples.values
        if f.ndim == 1:
            f = np.array([f]).T
        if self.parameters["addZerosStart"].value is not None and self.parameters["addZerosStart"].value is not False:
            f = np.concatenate((np.zeros([self.parameters["windowSize"].value, f.shape[1]]), f), axis=0)
            # if self.parameters["addZerosStart"].value == 0:
            #     f = np.concatenate((np.zeros([self.parameters["windowSize"].value, f.shape[1]]), f), axis=0)
            # else:
            #     f = np.concatenate((np.ones([self.parameters["windowSize"].value, f.shape[1]]) * f[0], f), axis=0)

        # Compute the self-similarity matrix
        S = self.compute_ssm(f, metric=self.parameters["distanceMetric"].value)

        # Compute gaussian kernel
        G = self.compute_gaussian_krnl(self.parameters["windowSize"].value)

        # Compute the novelty curve
        nc = self.compute_nc(S, G)

        result = copy.copy(samples)
        result.values = nc

        if self.parameters["addZerosStart"].value is not None and self.parameters["addZerosStart"].value is not False:
            result.values = result.values[self.parameters["windowSize"].value:]

        if self.parameters["debugViz"].value:
            self.plot(S, nc)
        return (result, )

    def plot(self, S, nc):

        import matplotlib.pyplot as plt

        plt.matshow(S)
        plt.plot(nc * (-100) + (len(nc) + 100))
        plt.show()


# cb = Checkerboard()
# x = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0],
#               [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]])
# print("self-similar", cb.compute_nc(x, np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])))
# print("cross-similar", cb.compute_nc(x, np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])))
# print("fll-kernell", cb.compute_nc(x, np.array([[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]])))
