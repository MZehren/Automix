"""
creates a transition based on the score of the transition
"""
import copy
import itertools
import math
from typing import List

import librosa
import numpy as np

from automix import rules
from automix.model.classes.deck import Deck
from automix.model.classes.track import Track
from automix.model.inputOutput.serializer.reaperProxy import Point
from automix.utils.quantization import diff
from automix.model.classes.signal import Signal

def generateMix(tracks, barOverlaps=[8]):
    """
    Try all the segments alignments and select the one giving the best score.
    """
    tempo = tracks[0].getTempo()
    trackA = tracks[0]
    trackA.position = 1
    trackA.jumpStart(trackA.getDownbeats()[0])
    markers = []
    finalTracks = [trackA]
    for i, trackB in enumerate(tracks[1:]):
        # Add the next track to the mix
        trackABoundaries = trackA.features["selectedPeaks"].times
        trackBBoundaries = trackB.features["selectedPeaks"].times
        trackB.synchronize(tempo)

        # Try all the alignments and keep the best one based on the rules
        bestAlignment = [-1, trackA, trackB, ""]
        for itA, itB, barOverlap in itertools.product(range(len(trackABoundaries)), range(len(trackBBoundaries)), barOverlaps):
            overlap = float(barOverlap) * 60 * 4 / tempo  # Overlap in seconds
            startA, switchA, stopA = getStartAndStop(trackABoundaries, itA, overlap)
            startB, switchB, stopB = getStartAndStop(trackBBoundaries, itB, overlap)
            finalA, finalB = generateFX(copy.deepcopy(trackA), startA, switchA, stopA, copy.deepcopy(trackB), startB, switchB,
                                        stopB)

            grade, logs = rules.runAll(
                createDecks([finalA, finalB]),
                boundaries=[finalB.getDeckTime(startB) - 10, finalA.getDeckTime(stopA) + 10],
                switches=[[finalB.getDeckTime(startB),
                           finalB.getDeckTime(switchB),
                           finalA.getDeckTime(stopA)]])

            if bestAlignment[0] <= grade:
                bestAlignment = (grade, finalA, finalB, logs, finalB.getDeckTime(switchB))

        # redo the good transition
        # TODO simplify
        if bestAlignment[0] != -1:
            finalTracks[-1] = bestAlignment[1]
            finalTracks.append(bestAlignment[2])
            trackA = bestAlignment[2]
            markers.append((bestAlignment[4], str(bestAlignment[3])))

    return createDecks(finalTracks), markers


def getStartAndStop(boundaries: List[float], i: int, crossfadeDuration: float):
    """
    Compute the previous and next position from the switch point i based on the list cue points and a maximal duration:
    The positions returned are either the closest cue point before and after the switch unlesss the distance exeeds the maximal duration
    """
    start = boundaries[i - 1] if i > 0 and boundaries[i] - boundaries[i - 1] < 4 * crossfadeDuration else max(
        [boundaries[i] - crossfadeDuration, 0])
    stop = boundaries[i + 1] if i + 1 < len(boundaries) and boundaries[
        i + 1] - boundaries[i] < 4 * crossfadeDuration else boundaries[i] + crossfadeDuration  # boundaries[-1]
    return start, boundaries[i], stop


def generateFX(trackA: Track, startA: float, switchA: float, endA: float, trackB: Track, startB: float, switchB: float,
               endB: float):
    """
    Create a transition between two tracks and start and stop segments
    startA, endA, starB, endB, are the track times of the crossfading
    TODO: factorize !
    """
    # Fine tune playback rate
    trackB.position = trackA.getDeckTime(
        switchA) - switchB / trackB.playRate  # B position = switch - distance in B from the switch
    trackA.length = (trackA.getDuration() - trackA.soffs) / trackA.playRate
    trackB.length = (trackB.getDuration() - trackB.soffs) / trackB.playRate
    startOverlap = trackB.getDeckTime(startB)
    endOverlap = trackA.getDeckTime(endA)
    fineTunePlaybackRate(trackA, startOverlap, endOverlap, trackB)

    # Fine tune position
    trackB.position = trackA.getDeckTime(
        switchA) - switchB / trackB.playRate  # B poition = switch - distance in B from the switch
    trackA.length = (trackA.getDuration() - trackA.soffs) / trackA.playRate
    trackB.length = (trackB.getDuration() - trackB.soffs) / trackB.playRate
    startOverlap = trackB.getDeckTime(startB)
    endOverlap = trackA.getDeckTime(endA)
    fineTunePosition(trackA, startOverlap, endOverlap, trackB)

    # normalize the volume
    # RG is the distance to -14 dB
    trackA.FX["gainPt"].append(Point(position=trackA.soffs, amplitude=trackA.getReplayGain() + 11))
    trackA.FX["gainPt"].append(Point(position=trackA.getDuration(), amplitude=trackA.getReplayGain() + 11))
    trackB.FX["gainPt"].append(Point(position=trackB.soffs, amplitude=trackB.getReplayGain() + 11))
    trackB.FX["gainPt"].append(Point(position=trackB.getDuration(), amplitude=trackB.getReplayGain() + 11))

    # crossfading
    trackA.FX["volPt"].append(Point(position=switchA, amplitude=0, curve=1))
    trackA.FX["volPt"].append(Point(position=endA, amplitude=-100, curve=1))
    trackB.FX["volPt"].append(Point(position=startB, amplitude=-100, curve=-1))
    trackB.FX["volPt"].append(Point(position=switchB, amplitude=0, curve=-1))

    # EQ correction
    # TODO: apply the gain before doing all that
    for i, band in enumerate(["lowPt", "highPt"]):  # ["lowPt", "midPt", "highPt"]
        # correction = frequencyCorrection(
        #     trackA.features["barBandsMSE"][i].getValues(trackA.getTrackTime(startOverlap), startA),
        #     trackB.features["barBandsMSE"][i].getValues(startB, endB),
        #     limit=np.max(trackA.features["barBandsMSE"][i].values))
        correction = -26
        trackB.FX[band].append(Point(position=startB, amplitude=correction, curve=1))
        trackB.FX[band].append(Point(position=switchB, amplitude=0, curve=1))

        # correction = frequencyCorrection(
        #     trackA.features["barBandsMSE"][i].getValues(startA, endA),
        #     trackB.features["barBandsMSE"][i].getValues(endB, trackB.getTrackTime(endOverlap)),
        #     limit=np.max(trackB.features["barBandsMSE"][i].values))
        trackA.FX[band].append(Point(position=switchA, amplitude=0, curve=-1))
        trackA.FX[band].append(Point(position=endA, amplitude=correction, curve=-1))

    return trackA, trackB


def fineTunePlaybackRate(trackA: Track, startOverlap: float, endOverlap: float, trackB: Track):
    """
    Look at the difference between all the beats in both tracks during the overlap. 
    fine tune the playbackrate from track B based on the mean difference between those two
    """
    window = 0.2
    trackABeats = [
        beat for beat in trackA.getDeckTimes(trackA.features["beats"].times)
        if beat > startOverlap - window and beat < endOverlap + window
    ]
    trackBBeats = [
        beat for beat in trackB.getDeckTimes(trackB.features["beats"].times)
        if beat > startOverlap - window and beat < endOverlap + window
    ]
    # newplaybackRate = np.sqrt(np.mean(np.square(np.diff(trackABeats)))) / np.sqrt(np.mean(np.square(np.diff(trackBBeats))))
    newplaybackRate = np.mean(np.diff(trackBBeats)) / np.mean(np.diff(trackABeats))
    if not math.isnan(newplaybackRate):
        trackB.playRate *= newplaybackRate

    return newplaybackRate, len(trackBBeats)


def fineTunePosition(trackA, startOverlap, endOverlap, trackB):
    """
    Look at the difference between all the beats in both tracks during the overlap.
    Fine tune the phase of track B to minimise the distance 
    """
    window = 0.2
    trackABeats = [
        beat for beat in trackA.getDeckTimes(trackA.features["beats"].times)
        if beat > startOverlap - window and beat < endOverlap + window
    ]
    trackBBeats = [
        beat for beat in trackB.getDeckTimes(trackB.features["beats"].times)
        if beat > startOverlap - window and beat < endOverlap + window
    ]
    deltas = diff(trackABeats, trackBBeats)
    averageShift = np.mean(deltas)
    trackB.position += averageShift
    # signalB = Signal(1, times=trackBBeats)
    # signalB.quantizeTo(Signal(1, times=trackABeats))
    # trackB.stretchMarkers = [(trackB.getTrackTime(trackABeats[i]), trackB.getTrackTime(trackBBeats[i]))
    #                          for i, _ in enumerate(trackABeats)]

    trackBBeats = [
        beat for beat in trackB.getDeckTimes(trackB.features["beats"].times) if beat > startOverlap and beat < endOverlap
    ]
    rmse = np.sqrt(np.mean(np.square(diff(trackABeats, trackBBeats))))
    return rmse


def frequencyCorrection(bandA, bandB, limit=2):
    """
    compute the correction to aply in dB to one signal to prevent it from hitting the limit threshold
    """
    correction = 1 - ((np.mean(bandA) + np.mean(bandB)) - limit)
    return min(librosa.amplitude_to_db([correction])[0], 0)


def createDecks(tracks):
    """
        group tracks in decks in a way to prevent overlap on one deck
    """
    decks = []
    for track in tracks:
        availableDecks = [deck for deck in decks if deck.tracks[-1].length + deck.tracks[-1].position < track.position]
        if not availableDecks:
            decks.append(Deck(name=len(decks), tracks=[]))
            availableDecks.append(decks[-1])

        availableDecks[0].tracks.append(track)
    [deck.updateFX() for deck in decks]
    return decks
