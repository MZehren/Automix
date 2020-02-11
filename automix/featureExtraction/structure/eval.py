import logging as log
from typing import List

import mir_eval
import numpy as np

from automix.model.classes.signal import Signal, SparseSegmentSignal


def evalCuesMutliple(
        Y_pred: List[Signal],
        Y_truth: List[Signal],
        window=0.5,
        averagePerDocument=False,  # TODO: rename this approach "mean" or "sum"
        returnDetails=False,
        limitSearchSpace=False,
        limitSearchNumber=False):
    """
    Get the metrics for an array of signals
    window = hit distance
    averagePerDocument = Return the document average instead of the sample average ("mean" vs "sum")
    returnDetails = return each individual score instead of the average
    limitSearchSpace = compute the hitRate only for the part annotated of the track. Stop at the last annotation.
    All the points detected after are not taken into account
    """

    result = {}
    if limitSearchSpace: # Remove all the prediction after the last annotation
        maxY = [max(Y_truth[i].times) if len(Y_truth[i].times) else 0 for i, _ in enumerate(Y_truth)]
        Y_pred = [Signal(1, times=[t for t in Y_pred[i].times if t <= maxY[i] + window]) for i, _ in enumerate(Y_pred)]
    
    if limitSearchNumber: # Remove all the predictions above the number of annotations in each track
        nY = [len(Y_truth[i]) for i, _ in enumerate(Y_truth)]
        Y_pred = [Signal(1, times=Y_pred[i].times[: nY[i]]) for i, _ in enumerate(Y_pred)]
        
    if averagePerDocument: # Average per documents: sum ?
        tracksResult = [evalCues(Y_pred[i], Y_truth[i], window=window) for i in range(len(Y_truth))]
        for field, value in tracksResult[0].items():
            if returnDetails:
                result[field] = [measure[field] for measure in tracksResult]
            else:
                result[field] = np.sum([measure[field] for measure in tracksResult]) / len(tracksResult)
    else: # Average per points: mean?
        # TODO: redundant call to hit
        precision = np.sum([hit(Y_pred[i], Y_truth[i], window)
                            for i, _ in enumerate(Y_truth)]) / np.sum(len(Y_pred[i]) for i, _ in enumerate(Y_truth))
        recall = np.sum([hit(Y_pred[i], Y_truth[i], window)
                         for i, _ in enumerate(Y_truth)]) / np.sum(len(Y_truth[i]) for i, _ in enumerate(Y_truth))
        fMeasure = 2 * (precision * recall) / (precision + recall)
        result = {"precision": precision, "recall": recall, "fMeasure": fMeasure}
        log.debug(result)
    return result


def evalCues(y_: Signal, y: Signal, window=0.5):
    """Get the F1, Precision and recall of estimated points (y_) to reference points (y)
    
    Args:
    ----
        y_ (Signal): [description]
        y (Signal): [description]
        window (float, optional): [description]. Defaults to 0.5.
    
    Returns:
    -------
        [type]: [description]
    """
    F, P, R = mir_eval.onset.f_measure(y.times, y_.times, window=window)
    return {"precision": P, "recall": R, "fMeasure": F}


def myEvalCues(y_: Signal, y: Signal, window=0.5):
    """
    return the precision, recall, f, in function of list of times.
    You can specify the minDistance 
    """
    if len(y_.times) == 0:
        return {"precision": 0, "recall": 0, "fMeasure": 0}

    # Probability that A is correct
    precision = hit(y_, y, window) / len(y_)
    # % of C we find
    recall = hit(y, y_, window) / len(y)
    if precision == 0 and recall == 0:
        fMeasure = 0
    else:
        fMeasure = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "fMeasure": fMeasure}


def hit(y_pred: Signal, y_truth: Signal, window: float):
    """ return the number of detection within a small window of an annotation.
    TODO (without counting an annotation mutilitple times)
    
    Args:
    ----
        y_ (Signal): Estimations
        y (Signal): Ground Thruth
        window (float): hit threshold
    
    Returns:
    -------
        [type]: [description]
    """
    if y_pred.times is None or y_truth.times is None:
        return 0

    if isinstance(y_truth, SparseSegmentSignal):
        return len([
            a for a in y_pred.times if a is not None
            and any([1 for i, _ in enumerate(y_truth.times) if a >= y_truth.times[i][0] - window and a <= y_truth.times[i][1] + window])
        ])
    else:
        return len([a for a in y_pred.times if a is not None and any([np.abs(a - c) < window for c in y_truth.times])])
