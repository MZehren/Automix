import copy
from typing import List

import librosa
import msaf.utils as U
import numpy as np
from scipy import signal
from scipy.ndimage import filters
from scipy.spatial import distance

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.featureExtraction.lowLevel import Normalize
from automix.model.classes.signal import Signal


class StructuralFeatures(Estimator):
    """
    Estimator computing the sf novelty from Serrà
    Implementation from msaf

    parameter_m_embedded = the number of samples to put together to make more sense of them -> it's similar to the window size
    parameter_M_gaussian = the std deviation of the gaussian filter / 2
    """

    def __init__(
            self,
            #  parameter_Mp_adaptive=28,
            #  parameter_offset_thres=0.05,
            parameter_M_gaussian=16,
            parameter_m_embedded=3,
            parameter_k_nearest=0.04,
            parameter_bound_norm_feats=np.inf,
            inputSamples="normalizedBarMSE",
            outputNovelty="noveltyMSE",
            cachingLevel=2,
            forceRefreshCache=False):
        self.parameters = {
            # "Mp_adaptive": Parameter(parameter_Mp_adaptive),
            # "offset_thres": Parameter(parameter_offset_thres),
            "M_gaussian": Parameter(parameter_M_gaussian),
            "m_embedded": Parameter(parameter_m_embedded),
            "k_nearest": Parameter(parameter_k_nearest),
            "bound_norm_feats": Parameter(parameter_bound_norm_feats)
        }
        self.inputs = [inputSamples]
        self.outputs = [outputNovelty]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, samples: Signal) -> Signal:
        """
        TODO
        """
        # Structural Features params
        # Mp = self.parameters["Mp_adaptive"].value  # Size of the adaptive threshold for
        # peak picking
        # od = self.parameters["offset_thres"].value  # Offset coefficient for adaptive
        # thresholding

        M = self.parameters["M_gaussian"].value  # Size of gaussian kernel in beats
        m = self.parameters["m_embedded"].value  # Number of embedded dimensions
        k = self.parameters["k_nearest"].value  # k*N-nearest neighbors for the
        # recurrence plot

        # Preprocess to obtain features, times, and input boundary indeces
        F = np.array(samples.values)
        if F.ndim == 1:
            F = np.array([F]).T

        if len(F.shape) == 2:
            F = np.concatenate((np.zeros((m // 2, F.shape[1])), F, np.zeros((m // 2, F.shape[1]))))
        else:
            F = np.concatenate((np.zeros(m // 2), F, np.zeros(m // 2)))
        # Normalize
        # F_norm = Normalize().predictOne(F)
        # F = U.normalize(F, norm_type=self.parameters["bound_norm_feats"].value)

        # Check size in case the track is too short
        if F.shape[0] > 20:

            # if self.framesync: # Whether to use frame-synchronous or beat-synchronous features.
            #     red = 0.1
            #     F_copy = np.copy(F)
            #     F = librosa.util.utils.sync(F.T, np.linspace(0, F.shape[0], num=F.shape[0] * red), pad=False).T

            # Emedding the feature space (i.e. shingle)
            # E[i] = F[i]+F[i+1]+F[i+2]
            E = embedded_space(F, m)
            # plt.imshow(E.T, interpolation="nearest", aspect="auto"); plt.show()

            # Recurrence matrix
            R = librosa.segment.recurrence_matrix(
                E.T,
                k=k * int(F.shape[0]),
                width=1,  # zeros from the diagonal
                metric="euclidean",
                sym=True).astype(np.float32)

            # Circular shift
            L = circular_shift(R)

            # Obtain structural features by filtering the lag matrix
            SF = gaussian_filter(L.T, M=M, axis=1)
            SF = gaussian_filter(L.T, M=1, axis=0)

            # Compute the novelty curve
            nc = compute_nc(SF)
            nc = nc[m//2:-m//2]
            times = samples.times[:-m]
            return (Signal(nc, times=times), )
        else:
            return (None, )


def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in range(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    return X


def gaussian_filter(X, M=8, axis=0):
    """Gaussian filter along the first axis of the feature matrix X."""
    for i in range(X.shape[axis]):
        if axis == 1:
            X[:, i] = filters.gaussian_filter(X[:, i], sigma=M / 2.)
        elif axis == 0:
            X[i, :] = filters.gaussian_filter(X[i, :], sigma=M / 2.)
    return X


def compute_gaussian_krnl(M):
    """Creates a gaussian kernel following Serra's paper."""
    g = signal.gaussian(M, M / 3., sym=True)
    G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
    G[M // 2:, :M // 2] = -G[M // 2:, :M // 2]
    G[:M // 2, M // 1:] = -G[:M // 2, M // 1:]
    return G


def compute_ssm(X, metric="seuclidean"):
    """Computes the self-similarity matrix of X."""
    D = distance.pdist(X, metric=metric)
    D = distance.squareform(D)
    D /= float(D.max())
    return 1 - D


def compute_nc(X):
    """Computes the novelty curve from the structural features."""
    N = X.shape[0]
    # nc = np.sum(np.diff(X, axis=0), axis=1) # Difference between SF's

    nc = np.zeros(N)
    for i in range(N - 1):
        nc[i] = distance.euclidean(X[i, :], X[i + 1, :])

    # Normalize
    nc += np.abs(nc.min())
    nc /= float(nc.max())
    return nc


def pick_peaks(nc, L=16, offset_denom=0.1):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = nc.mean() * float(offset_denom)
    th = filters.median_filter(nc, size=L) + offset
    # th = filters.gaussian_filter(nc, sigma=L/2., mode="nearest") + offset
    # import pylab as plt
    # plt.plot(nc)
    # plt.plot(th)
    # plt.show()
    # th = np.ones(nc.shape[0]) * nc.mean() - 0.08
    peaks = []
    for i in range(1, nc.shape[0] - 1):
        # is it a peak?
        if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            # is it above the threshold?
            if nc[i] > th[i]:
                peaks.append(i)
    return peaks


def circular_shift(X):
    """Shifts circularly the X squre matrix in order to get a
        time-lag matrix."""
    N = X.shape[0]
    L = np.zeros(X.shape)
    for i in range(N):
        L[i, :] = np.asarray([X[(i + j) % N, j] for j in range(N)])
    return L


def embedded_space(X, m, tau=1):
    """Time-delay embedding with m dimensions and tau delays."""
    N = X.shape[0] - int(np.ceil(m))
    Y = np.zeros((N, int(np.ceil(X.shape[1] * m))))
    for i in range(N):
        # print X[i:i+m,:].flatten().shape, w, X.shape
        # print Y[i,:].shape
        rem = int((m % 1) * X.shape[1])  # Reminder for float m
        Y[i, :] = np.concatenate((X[i:i + int(m), :].flatten(), X[i + int(m), :rem]))
    return Y
