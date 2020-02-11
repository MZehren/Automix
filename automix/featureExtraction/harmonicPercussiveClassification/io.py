import datetime as dt
import numpy as np
import pandas as pd

from model.classes.signal import Signal


def readAnnotations(pathCSV, downbeats, delimiter=","):
    """Reads in harmonic and percussive annotations from a csv file for a song.

    Args:
        pathCSV (str): The path to the csv file containing annotations regarding
            harmonicness and percussiveness of the segments of the song.
        downbeats (list of float): The times where downbeats occur in the song.
        delimiter (str): The delimiter used in the csv file.

    Returns:
        tuple of Signal: A harmonic and a percussive signal containing the
        annotations for each bar of the song.

    """

    # Parse annotations from csv file
    def convertTime(timeStr):
        try:
            timeObj = dt.datetime.strptime(timeStr, "%M:%S")
        except ValueError:
            timeObj = dt.datetime.strptime(timeStr, "%S")
        return dt.timedelta(
            minutes=timeObj.minute, seconds=timeObj.second).total_seconds()

    annotationsDataFrame = pd.read_csv(
        pathCSV,
        sep=delimiter,
        converters={
            0: convertTime,
            1: convertTime,
            2: int,
            3: int
        },
        skipinitialspace=True)
    annotations = annotationsDataFrame.values

    # Convert annotations for segments to annotations for bars
    harmonicAnnotated = []
    percussiveAnnotated = []
    for i in range(len(downbeats) - 1):
        barStart = downbeats[i]
        barEnd = downbeats[i + 1]
        candidate = (0, False, False)  # overlap, harmonic, percussive
        for segmentStart, segmentEnd, percussive, harmonic in \
                annotations[:, :4]:
            overlap = min(barEnd, segmentEnd) - max(barStart, segmentStart)
            if overlap > candidate[0]:
                candidate = (overlap, harmonic, percussive)
        harmonicAnnotated.append(candidate[1])
        percussiveAnnotated.append(candidate[2])

    return Signal(harmonicAnnotated, times=downbeats[:-1]), \
           Signal(percussiveAnnotated, times=downbeats[:-1])


def writeFittingResults(pathCSV, matrix, header, delimiter=","):
    """Writes a result matrix of scores and parameter values to a csv file.

    Args:
        pathCSV (str): Path to the output csv file.
        matrix (2d np.array): The result matrix.
        header (list of str): The headers for the columns of the csv file.
        delimiter (str): The delimiter used in the csv file.

    """

    dataFrame = pd.DataFrame(matrix)
    dataFrame.to_csv(
        pathCSV,
        sep=delimiter,
        float_format="%.2f", # doesn't work?
        header=header,
        index=False)