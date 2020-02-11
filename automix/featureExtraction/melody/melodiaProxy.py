import utils

import vamp
import librosa
import numpy as np


def extractMelodie(path, grid, minSamplesRatio=0.5):
    data, rate = Track.readFile(path)

    # The pitch of the main melody. Each row of the output contains a timestamp and the corresponding frequency of the melody in Hertz.
    # Non-voiced segments are indicated by zero or negative frequency values.
    # Negative values represent the algorithm's pitch estimate for segments estimated as non-voiced,
    # in case the melody is in fact present there.
    result = vamp.collect(data, rate, "mtg-melodia:melodia")
    melodieSampleRate = float(len(result['vector'][1])) / (
        float(len(data)) / rate)

    #Beat samples
    beatMelodieSamples = [
        result['vector'][1][int(grid[tickI] * melodieSampleRate):int((
            grid[tickI + 1]) * melodieSampleRate)]
        for tickI in range(len(grid) - 1)
    ]
    beatPositiveSamples = [[sample for sample in samples if sample > 0]
                           for samples in beatMelodieSamples]
    # beatNote = [(np.mean(samples)) if len(samples) >= minNumSamples else "-1"
    #             for samples in beatPositiveSamples]

    beatNote = [
        utils.hertzToNote(np.percentile(samples, [20, 50, 80])[1]) if
        len(samples) >= minSamplesRatio * len(beatMelodieSamples[i]) else "-1"
        for i, samples in enumerate(beatPositiveSamples)
    ]
    return beatNote
