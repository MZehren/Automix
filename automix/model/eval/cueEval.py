"""
Do the peakpicking and compute the metrics
"""
import numpy as np

from automix.featureExtraction.lowLevel import PeakPicking
from automix.model.classes import Signal


def getMetricsSparse(A, C, minDistance=0.5):
    """
    return the precision, recall, f, in function of list of times.
    You can specify the minDistance 
    """
    if not A:
        return {"precision": 0, "recall": 0, "fMeasure": 0}

    precision = len([a for a in A if any([np.abs(a - c) < minDistance for c in C])]) / len(A)  #probability that A is correct
    recall = len([c for c in C if any([np.abs(a - c) < minDistance for a in A])]) / len(C)  #% of C we find
    fMeasure = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "fMeasure": fMeasure}


def getMetricsDense(dfA, dfC):
    """
    Return metrics from binary list such as [0,1,0], [1,1,0] where the index should be the same events
    dfA are the samples where the antecedent are True or False
    dfC are the samples where the consequents are True or False 
    """
    support = dfA.sum()
    precision = (dfA & dfC).sum() / dfA.sum()  #P(A&C)/P(A)    or     P(C|A)
    recall = (dfA & dfC).sum() / dfC.sum()
    lift = precision / (dfC.sum() / len(dfA))
    fMeasure = 2 * (precision * recall / (precision + recall))
    return {
        "support": round(support, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "lift": round(lift, 2),
        "fMeasure": round(fMeasure, 2)
    }


def findPhaseLocal(signal: Signal, grid: Signal, period: int, toleranceWindow=0):
    """
    find the phase of the signal based on it's amplitude at the grid positions and the number of peaks
    - signal: works best with a discrete signal as no aglomeration is done
    - grid: positions of the beats
    - period: the periodicity to test
    - tolerance window: if not at 0, returns the closest value in the signal to the grid, within the tolerance window
    
    test:
    # result = findPhase(Signal(np.ones(5), times=np.array([0, 4, 8, 9, 12])+1), Signal(np.ones(16), times=range(16)), period=4)
    # print(result) = 1
    """
    phases = []
    for phase in range(period):
        values = [signal.getValue(grid.times[i], toleranceWindow=0.1) for i in range(phase, len(grid), period)]
        values = [v for v in values if v is not None]
        phases.append((np.sum(values) * len(values)))

    bestPhase = np.argmax(phases)
    return bestPhase


def getPhase(track, features, period):
    """
    Get the phase of the track depending on all the features specified and the period
    """
    from automix.utils.quantization import clusterValues

    phasePerFeature = []
    for feature in features:
        phasePerFeature.append(findPhaseLocal(feature, track.features["strongBeats"], period=period))
    counts = np.bincount(phasePerFeature)
    #     print(phases, counts, np.argmax(counts))
    return np.argmax(counts)


def getScore(features,
             aggregation="independant",
             relativeThreshold=0.3,
             top=3,
             returnCues=False,
             minDistancePeak=32,
             minDistanceCluster=0,
             period=2):
    """
    Compute for each track and each feature provided, the top k peaks. Then aggregate them to compute the score,
     - Feature top: use only the n first peaks of each feature
     - returnCues: Return the peaks on not the score
     - minDistance: aggregate the peaks under minDistance
     - aggregation: Concatenate the peaks. Can either extract the peaks from all the features independently (independant)
     Or take the peaks in the cruve after multiplying all the features element wise (multiply)
    """
    from automix.utils.quantization import clusterValues, findPhase, quantize

    cues = []
    gtCues = []
    result = {}

    for i, track in enumerate(tracks):
        # for feature in features:
        #     #Snap the peaks to the closest strong beat
        #     newCues = quantize(track.features["strongBeats"], newCues)

        #Concatenate all the peaks from all the features
        peakSignals = []
        pp = PeakPicking(parameterMinDistance=minDistancePeak, parameterRelativeThreshold=relativeThreshold)
        if aggregation == "independant":
            peakSignals = [pp.predictOne(track.features[feature])[0] for feature in features]
            newCues = np.concatenate([signal.times for signal in peakSignals])

        elif aggregation == "multiply":
            #If the aggregation is set to multipl, the features shouldn't be the peaks but the novelty
            newCurve = np.ones(len(track.features[features[0]].values))
            for feature in features:
                newCurve = np.multiply(newCurve, track.features[feature].values)
            peakSignals = pp.predictOne(Signal(newCurve, times=track.features[features[0]].times))
            newCues = peakSignals[0].times

        for cue in newCues:
            if cue not in track.features["strongBeats"].times:
                print("cue not on strongbeat", cue)

        # #Snap the peaks to the closest strong beat
        # newCues = quantize(track.features["strongBeats"], newCues)

        #Cluster the peaks to remove close outliers
        if minDistanceCluster:
            newCues = clusterValues(newCues, minDistance=minDistanceCluster)
        else:
            newCues = list(set(newCues))
        newCues.sort()

        #Identify the beat-period
        if period and newCues:
            phase = getPhase(track, peakSignals, period)
            inPhase = track.features["strongBeats"].times[phase:-1:period]
            newCues = [cue for cue in newCues if cue in inPhase]

        firstK = newCues[:top]
        cues += firstK

        result[track.name] = {
            "cues": firstK,  #Cues candidates
            "cuesFeature": {
                features[j]: len([1 for t in signal.times if t in firstK]) / len(firstK) if len(firstK) else 0
                for j, signal in enumerate(peakSignals)
            },
        }

        if any(gttracks):
            gtCues += gttracks[i].features["boundaries"]
            result[track.name]["gtCues"] = gttracks[i].features["boundaries"]  #Cuesannotated
            result[track.name]["gtCuesFeature"] = {
                features[j]: len([
                    1 for t in signal.times
                    if t in firstK and any([np.abs(t - gtT) <= 0.5 for gtT in gttracks[i].features["boundaries"]])
                ]) / len(gttracks[i].features["boundaries"])
                for j, signal in enumerate(peakSignals)
            }

    if returnCues:
        return result
    result = getMetricsSparse(cues, gtCues)
    #     print(len(cues), len(gtCues), result)
    return result
