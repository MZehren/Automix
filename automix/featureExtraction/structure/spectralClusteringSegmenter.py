#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Music segmentation using timbre, pitch, repetition and time.

If run as a program, usage is:

    ./segmenter.py AUDIO.mp3 OUTPUT.lab

'''

import sys
import os
import argparse
import string

import numpy as np
import scipy.spatial
import scipy.signal
import scipy.linalg

import sklearn.cluster

# Requires librosa-develop 0.3 branch
import librosa

# Suppress neighbor links within REP_WIDTH beats of the current one
REP_WIDTH = 3

# Only consider repetitions of at least (FILTER_WIDTH-1)/2
FILTER_WIDTH = 1 + 2 * 8

# How much state to use?
N_STEPS = 2

# Local model
N_MELS = 128
N_MFCC = 13

# Which similarity metric to use?
METRIC = 'sqeuclidean'

# Sample rate for signal analysis
SR = 22050

# Hop length for signal analysis
HOP_LENGTH = 512

# Maximum number of structural components to consider
MAX_REP = 10

# Minimum and maximum average segment duration
MIN_SEG = 10.0
MAX_SEG = 30.0

# Minimum tempo threshold.
# If we dip below this, double the starting tempo and try again
MIN_TEMPO = 70.0

# Minimum duration (in beats) of a "non-repeat" section
MIN_NON_REPEATING = (FILTER_WIDTH - 1) / 2

# Pre-compute label identifiers for the detected segments
SEGMENT_NAMES = list(string.ascii_uppercase)
for x in string.ascii_uppercase:
    SEGMENT_NAMES.extend(['%s%s' % (x, y) for y in string.ascii_lowercase])


def get_beats(y, sr, hop_length):
    '''Extract beats from an audio time series.

    :parameters:
        - y : np.ndarray [shape=(n,)]
          Audio time series

        - sr : int > 0
          Sampling rate of y

        - hop_length : int > 0
          Number of samples to advance in each frame

    :returns:
        - tempo : float > 0
          Tempo in beats per minute

        - beats : np.ndarray [shape=(m,)]
          Positions (frame numbers) of detected beat events
    '''

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median, n_mels=128)

    bpm, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length, start_bpm=120)

    if bpm < MIN_TEMPO:
        bpm, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length, bpm=2 * bpm)

    return bpm, beats


def features(filename):
    '''Extract feature data for spectral clustering segmentation.

    :parameters:
        - filename : str
            Path on disk to an audio file

    :returns:
        - X_cqt : np.ndarray [shape=(d1, n)]
          A beat-synchronous log-power CQT matrix

        - X_timbre : np.ndarray [shape=(d2, n)]
          A beat-synchronous MFCC matrix

        - beat_times : np.ndarray [shape=(n, 2)]
          Timing of beat intervals
    '''
    print('\t[1/5] loading audio')
    y, sr = librosa.load(filename, sr=SR)
    # y = librosa.resample(y, sr, SR, res_type='sinc_fastest')
    sr = SR

    print('\t[2/5] Separating harmonic and percussive signals')
    y_harm, y_perc = librosa.effects.hpss(y)

    print('\t[3/5] detecting beats')
    bpm, beats = get_beats(y=y_perc, sr=sr, hop_length=HOP_LENGTH)

    print('\t[4/5] generating CQT')
    X_cqt = librosa.cqt(y=y_harm, sr=sr, hop_length=HOP_LENGTH, bins_per_octave=12, fmin=librosa.midi_to_hz(24), n_bins=72)

    # Compute log CQT power
    X_cqt = librosa.amplitude_to_db(np.abs(X_cqt), ref=np.max)

    # Compute MFCCs
    print('\t[5/5] generating MFCC')
    X_melspec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP_LENGTH, n_mels=N_MELS)

    X_timbre = librosa.feature.mfcc(S=librosa.power_to_db(X_melspec), n_mfcc=N_MFCC)

    # Resolve any timing discrepancies due to CQT downsampling
    n = min(X_cqt.shape[1], X_timbre.shape[1])

    # Trim the beat detections to fit within the shape of X*
    beats = beats[beats < n]

    # Pad on a frame=0 beat for synchronization purposes
    beats = np.unique(np.concatenate([[0], beats]))

    # Convert beat frames to beat times
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)

    # Take on an end-of-track marker.  This is necessary if we want
    # the output intervals to span the entire track.
    beat_times = np.concatenate([beat_times, [float(len(y)) / sr]])

    beat_intervals = np.asarray(list(zip(beat_times[:-1], beat_times[1:])))

    # Synchronize the feature matrices
    X_cqt = librosa.util.sync(X_cqt, beats, aggregate=np.median)
    X_timbre = librosa.util.sync(X_timbre, beats, aggregate=np.mean)

    return X_cqt, X_timbre, beat_intervals


def save_segments(outfile, boundaries, beat_intervals, labels=None):
    '''Save detected segments to a .lab file.

    :parameters:
        - outfile : str
            Path to output file

        - boundaries : list of int
            Beat indices of detected segment boundaries

        - beat_intervals : np.ndarray [shape=(n, 2)]
            Intervals of beats

        - labels : None or list of str
            Labels of detected segments
    '''

    if labels is None:
        labels = [('Seg#%03d' % idx) for idx in range(1, len(boundaries))]

    times = [beat_intervals[interval, 0] for interval in boundaries[:-1]]
    times.append(beat_intervals[-1, -1])

    with open(outfile, 'w') as f:
        for idx, (start, end, lab) in enumerate(zip(times[:-1], times[1:], labels), 1):
            f.write('%.3f\t%.3f\t%s\n' % (start, end, lab))

    pass


def get_num_segs(duration, min_seg, max_seg, k_min=2, k_max=3):
    '''Estimate bounds on the number of segments from a track duration.

    :parameters:
        - duration : float > 0
          Track duration

        - min_seg : float > 0
          Minimum average segment duration

        - max_seg : float > 0
          Maximum average segment duration

        - k_min : int > 0
          Lower bound on k_min

        - k_max : int > 0
          Lower bound on k_max

    :returns:
        - k_min : int >= 2
          Minimum number of segments

        - k_max : int > k_min
          Maximum number of segments
    '''

    k_min = max(k_min, np.floor(duration / max_seg).astype(int))
    k_max = max(k_max, np.ceil(duration / min_seg).astype(int))

    return k_min, k_max


def median_padded(S, width):
    '''
    Apply horizontal median filter to a matrix S with reflection padding.

    This prevents winnowing of values at the left- and right-edges of S.

    :parameters:
        - S : np.ndarray [shape=(d, n)]
          The input matrix to filter

        - width : int > 0
          The width of the median filter

    :returns:
        - S_filtered : np.ndarray [shape=S.shape]
          The filtered matrix
    '''

    # First, make a reflection-padded copy of S
    Sf = np.pad(S, [(0, 0), (width, width)], mode='reflect')

    # Then, apply the median filter
    Sf = scipy.signal.medfilt2d(Sf, kernel_size=(1, width))

    # Finally, trim the padding
    Sf = Sf[:, width:-width]
    return Sf


def sym_laplacian(A):
    '''Compute the symmetric, normalized graph laplacian.

    :parameters:
      - A : np.ndarray >= 0 [shape=(n, n)]
        A square, symmetric, non-negative matrix containing
        affinities between items.

    :returns:
      - L : np.ndarray [shape=(n, n)]
        The laplacian of A
    '''

    Dinv = np.sum(A, axis=1)**-1.0

    Dinv[~np.isfinite(Dinv)] = 1.0

    Dinv = np.diag(Dinv**0.5)

    L = np.eye(len(A)) - Dinv.dot(A.dot(Dinv))

    return L


def combine_graphs(A_rep, A_loc):
    ''' Find mu such that

        mu * deg(A_rep, i) ~= (1-mu) * deg(A_loc, i)

    This attempts to produce a weighted combination of A_rep and A_loc
    such that a random walk on the combined graph has approximately equal
    probability of transitioning through an edge from A_loc or A_rep.

    :parameters:
      - A_loc : np.ndarray >= 0 [shape=(n, n)]
        A square, symmetric, non-negative matrix containing the
        local path affinities.

      - A_rep : np.ndarray >= 0 [shape=(n, n)]
        A square, symmetric, non-negative matrix containing the
        repetition affinities

    :returns:
      - A : np.ndarray >= 0 [shape=(n, n)]
        The optimally weighted combination of A_rep and A_loc:
          A = mu * A_rep + (1 - mu) * A_loc
    '''
    d1 = np.sum(A_rep, axis=1)
    d2 = np.sum(A_loc, axis=1)

    ds = d1 + d2
    mu = d2.dot(ds) / np.dot(ds, ds)
    return mu * A_rep + (1 - mu) * A_loc


def factorize(L, k=20):
    '''Factorize a square matrix L into eigenvalues.

    :parameters:
        - L : np.ndarray [shape=(n,n)]
          The matrix to be factorized

        - k : int > 0
          The maximum number of factors to return

    :returns:
        - e_vecs : np.ndarray [shape=(min(n, k), n)]
          The eigenvectors of L as rows, sorted ascending
          by corresponding eigenvalue
    '''
    e_vals, e_vecs = scipy.linalg.eig(L)
    e_vals = e_vals.real
    e_vecs = e_vecs.real
    idx = np.argsort(e_vals)

    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]

    if len(e_vals) < k + 1:
        k = -1

    return e_vecs[:, :k].T


def label_rep_sections(X, boundaries, n_types):
    '''Label repeated sections.

    This is used to condense point-wise labels down to segment-wise labels.

    Each segment's centroid is computed, and the centroids are clustered 
    to produce labels.

    :parameters:
        - X : np.ndarray [shape=(d, n)]
          Feature matrix (eg latent factors)

        - boundaries : np.ndarray
          Detected segment boundaries

        - n_types : int > 0
          Number of segment types

    :returns:
        - intervals : list of int
          Detected segment intervals

        - labels : list of str
          Label for each segment
    '''

    # Classify each segment centroid
    X_rep_stack = librosa.util.sync(X, boundaries)

    C = sklearn.cluster.KMeans(n_clusters=n_types, tol=1e-8)

    labels = C.fit_predict(X_rep_stack.T)

    return zip(boundaries[:-1], boundaries[1:]), labels


def fixed_partition(Lf, n_types):
    '''Cluster data using a fixed number of partitions.

    :parameters:
        - Lf : np.ndarray [shape=(d, n)]
          Latent factors

        - n_types : int > 0, <= d
          Number of segment types to allow.

    :returns:
        - boundaries : np.ndarray [shape=(m,)]
          Indices of segment boundaries

        - labels : np.ndarray [shape=(m,)]
          Label indices of segment boundaries
    '''

    # Build the affinity matrix on the first n_types-1 repetition features
    Y = librosa.util.normalize(Lf[:n_types].T, norm=2, axis=1)

    # Try to label the data with n_types
    C = sklearn.cluster.KMeans(n_clusters=n_types, tol=1e-10, n_init=100)
    labels = C.fit_predict(Y)

    boundaries = 1 + np.asarray(np.where(labels[:-1] != labels[1:])).reshape((-1, ))

    boundaries = np.unique(np.concatenate([[0], boundaries, [len(labels)]]))

    intervals, labels = label_rep_sections(Y.T, boundaries, n_types)

    return boundaries, labels


def label_entropy(labels):
    '''Estimate the entropy of an array of labels.

    The underlying distribution is assumed to be iid multinomial.

    :parameters:
        - labels : np.array [shape=(n,)]
            Array of labels

    :returns:
        - entropy : float >= 0
            Entropy (in nats) of labels
    '''
    values = np.unique(labels)
    hits = np.zeros(len(values))

    for v in values:
        hits[v] = np.sum(labels == v)

    hits = hits / hits.sum()

    return scipy.stats.entropy(hits)


def label_clusterer(Lf, k_min, k_max):
    '''Automatically estimate the partition using maximum entropy labeling.

    The number of component types if varied from k_min to k_max,
    and the one which achieves highest label entropy (under an assumed 
    iid multinomial assumption) while satisfying minimum average segment
    durations is selected.

    :parameters:
        - Lf : np.ndarray [shape=(d, n)]
          Latent factors

        - k_min : int >= 2
          Minimum number of segment types

        - k_max : int > k_min
          Maximum number of segment types

    :returns:
        - boundaries : np.ndarray [shape=(m,)]
          Indices of segment boundaries

        - labels : np.ndarray [shape=(m,)]
          Label indices of segment boundaries
    '''

    best_score = -np.inf
    best_boundaries = [0, Lf.shape[1]]
    best_n_types = 1
    Y_best = Lf[:1].T

    label_dict = {}

    # The trivial solution
    label_dict[1] = np.zeros(Lf.shape[1])

    for n_types in range(2, 1 + len(Lf)):
        Y = librosa.util.normalize(Lf[:n_types].T, norm=2, axis=1)

        # Try to label the data with n_types
        C = sklearn.cluster.KMeans(n_clusters=n_types, n_init=100)
        labels = C.fit_predict(Y)
        label_dict[n_types] = labels

        # Find the label change-points
        boundaries = 1 + np.asarray(np.where(labels[:-1] != labels[1:])).reshape((-1, ))

        boundaries = np.unique(np.concatenate([[0], boundaries, [len(labels)]]))

        # boundaries now include start and end markers; n-1 is the number of segments
        feasible = (len(boundaries) > k_min)

        score = label_entropy(labels) / np.log(n_types)

        if score > best_score and feasible:
            best_boundaries = boundaries
            best_n_types = n_types
            best_score = score
            Y_best = Y

    # Did we fail to find anything with enough boundaries?
    # Take the last one then
    if best_boundaries is None:
        best_boundaries = boundaries
        best_n_types = n_types
        Y_best = librosa.util.normalize(Lf[:best_n_types].T, norm=2, axis=1)

    intervals, best_labels = label_rep_sections(Y_best.T, best_boundaries, best_n_types)

    return best_boundaries, best_labels


def estimate_bandwidth(D, k):
    '''Estimate the bandwidth of a gaussian kernel.

    sigma is computed as the average distance between 
    each point and its kth nearest neighbor.

    :parameters:
        - D : np.ndarray [shape=(n, n)]
          A squared euclidean distance matrix

        - k : int > 0
          Number of neighbors to use

    :returns:
        - sigma : float > 0
          Estimated bandwidth
    '''

    D_sort = np.sort(D, axis=1)

    if 1 + k >= len(D):
        k = len(D) - 2

    sigma = np.mean(D_sort[:, 1 + k])
    return sigma


def self_similarity(X, k):
    '''Construct a self-similarity matrix from a feature matrix.

    :parameters:
        - X : np.ndarray [shape=(d, n)]
          Feature matrix, each column corresponds to one sample

        - k : int > 0
          Number of nearest neighbors to use when estimating the
          kernel bandwidth

    :returns:
        - A = np.ndarray [shape=(n, n)]
          Gaussian kernel matrix
    '''

    D = scipy.spatial.distance.cdist(X.T, X.T, metric=METRIC)
    sigma = estimate_bandwidth(D, k)
    A = np.exp(-0.5 * (D / sigma))
    return A


def lsd(X_rep, X_loc, beat_intervals, parameters):
    '''Laplacian structural decomposition.

    :parameters:
        - X_rep : np.ndarray [shape=(d1, n)]
          Features to be used for generating repetition links

        - X_loc : np.ndarray [shape=(d2, n)]
          Features to be used for generating local path links

        - beat_interval : np.ndarray [shape=(n, 2)]
          Array of beat interval timings

        - parameters : dict
          Parameter dictionary as constructed by process_arguments()
    '''

    # Find the segment boundaries
    print('\tpredicting segments...')
    k_min, k_max = get_num_segs(beat_intervals[-1, -1], MIN_SEG, MAX_SEG)

    # Get the raw recurrence plot
    X_rep_pad = np.pad(X_rep, [(0, 0), (N_STEPS, 0)], mode='edge')
    X_rep_stack = librosa.feature.stack_memory(X_rep_pad, n_steps=N_STEPS)[:, N_STEPS:]

    # Compute the number of nearest neighbor links to generate
    k_link = 1 + int(np.ceil(2 * np.log2(X_rep.shape[1])))

    # Generate the repetition kernel
    A_rep = self_similarity(X_rep_stack, k=k_link)

    # And the timbre similarity kernel
    A_loc = self_similarity(X_loc, k=k_link)

    # Build the harmonic recurrence matrix
    recurrence = librosa.segment.recurrence_matrix(X_rep_stack, k=k_link, width=REP_WIDTH, metric=METRIC,
                                                   sym=True).astype(np.float32)

    # filter the recurrence plot by diagonal majority vote
    lag_matrix = librosa.segment.recurrence_to_lag(recurrence)
    lag_filtered = median_padded(lag_matrix, FILTER_WIDTH)
    rec_filtered = librosa.segment.lag_to_recurrence(lag_filtered)

    # Symmetrize the filtered matrix
    rec_filtered = np.maximum(rec_filtered, rec_filtered.T)

    # Suppress the main diagonal
    rec_filtered[np.diag_indices_from(rec_filtered)] = 0

    # We can jump to a random neighbor, or +- 1 step in time
    # Call it the infinite jukebox matrix
    A_combined = combine_graphs(rec_filtered * A_rep, (np.eye(len(A_loc), k=1) + np.eye(len(A_loc), k=-1)) * A_loc)

    # Get the graph laplacian
    L = sym_laplacian(A_combined)

    # Get the bottom k eigenvectors of L
    L_factors = factorize(L, k=1 + MAX_REP)

    # TODO:   2014-11-01 08:44:54 by Brian McFee <brian.mcfee@nyu.edu>
    #   probably should return here, pick up the partition selection elsewhere

    if parameters['num_types']:
        boundaries, labels = fixed_partition(L_factors, parameters['num_types'])
    else:
        boundaries, labels = label_clusterer(L_factors, k_min, k_max)

    # Output lab file
    return (boundaries, beat_intervals, labels)
    # print '\tsaving output to ', parameters['output_file']
    # save_segments(parameters['output_file'],
    #               boundaries,
    #               beat_intervals,
    #               labels)


def process_arguments(args):
    '''Argument parser.

    Returns a dictionary of parameters extracted from the command line.
    '''

    parser = argparse.ArgumentParser(description='Music segmentation')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False, help='verbose output')

    parser.add_argument('-m',
                        '--num-types',
                        dest='num_types',
                        action='store',
                        type=int,
                        default=None,
                        help='fix the number of segment types')

    parser.add_argument('input_song', action='store', help='path to input audio data')

    parser.add_argument('output_file', action='store', help='path to output segment file')

    return vars(parser.parse_args(args))


# Main block here
if __name__ == '__main__':

    parameters = process_arguments(sys.argv[1:])

    # Load the features
    print('- ', os.path.basename(parameters['input_song']))
    X_cqt, X_timbre, beat_intervals = features(parameters['input_song'])

    lsd(X_cqt, X_timbre, beat_intervals, parameters)
