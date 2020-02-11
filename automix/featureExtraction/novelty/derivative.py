import copy
from scipy import signal
from typing import List

import findiff
import librosa
import numpy as np

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class Derivative(Estimator):
    """
    Estimator computing the derivative of given discrete signal.
    """

    def __init__(self,
                 parameterWindowSize=8,
                 parameterAbsoluteDiff=True,
                 parameterGaussianCoef=5,
                 inputSamples="normalizedBarMSE",
                 outputNovelty="noveltyMSE",
                 cachingLevel=2,
                 forceRefreshCache=False):
        self.parameters = {
            "windowSize": Parameter(parameterWindowSize),
            "absoluteDiff": Parameter(parameterAbsoluteDiff),
            "gaussianCoef": Parameter(parameterGaussianCoef)
        }
        self.inputs = [inputSamples]
        self.outputs = [outputNovelty]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, samples: Signal) -> Signal:
        """
        Compute the central finite difference of the form: (-f(x - h)/2 + f(x + h)/2) / h
        see: https://en.wikipedia.org/wiki/Finite_difference_coefficient#cite_note-fornberg-1
        """
        f = samples.values
        W, offsets = self.getCenteredWeights(self.parameters["windowSize"].value, std=self.parameters["gaussianCoef"].value)

        difference = [
            np.sum([f[x + offset] * W[i] for i, offset in enumerate(offsets) if x + offset >= 0 and x + offset < len(f)])
            for x, _ in enumerate(f)
        ]
        # if len(f.shape) == 2:
        #     paddedF = np.concatenate((np.zeros((window // 2, f.shape[1])), f, np.zeros((window // 2, f.shape[1]))))
        # else:
        #     paddedF = np.concatenate((np.zeros(window // 2), f, np.zeros(window // 2)))

        # X = range(window // 2, len(paddedF) - window // 2)
        # difference = [np.sum([paddedF[x + i - window // 2] * w for i, w in enumerate(W)], axis=0) / window for x in X]
        # differenceOld = [(np.sum(f[i:i + window // 2]) - np.sum(f[max(0, i - window // 2):i])) / (window)
        #                  for i, _ in enumerate(f)]

        if self.parameters["absoluteDiff"].value:
            difference = np.abs(difference)

        result = copy.copy(samples)
        result.values = difference
        return (result, )

    def getCenteredWeights(self, window, coefficients="gaussian", std=5):
        """
        returns the coefficient and the indexes
        """
        offsets = range(-window // 2, window // 2)

        if coefficients == "findiff":
            coefs = findiff.coefficients(1, window)["center"]
            return (coefs["coefficients"], coefs["offsets"])
        elif coefficients == "gaussian":
            coefs = signal.gaussian(window, std, sym=True)  # get the guaussien filters
            coefs = coefs * 2 / np.sum(coefs)  # scale them to sum to one for each side
            coefs = [coef if i >= window // 2 else -coef for i, coef in enumerate(coefs)]  # invert the first weights
            return coefs, offsets
        else:
            coefs = [1 / window if i >= window // 2 else -1 / window for i in range(window)]
            offsets = [range(-window // 2, window // 2)]
            return (coefs, offsets)


# import matplotlib.pyplot as plt

# d = Derivative()

# w1, _ = d.getCenteredWeights(16, std=4)
# plt.plot(_, w1, label="std 4")
# w1, _ = d.getCenteredWeights(16, std=100)
# plt.plot(_, w1, label="std 100")
# w1, _ = d.getCenteredWeights(16, std=16//3)
# plt.plot(_, w1, label="std 16/3")

# plt.legend()
# plt.show()