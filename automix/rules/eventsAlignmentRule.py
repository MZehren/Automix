import numpy as np

from automix.rules.rule import Rule
from automix.utils import quantization


class EventsAlignmentRule(Rule):
    """
    The structure of the tracks should be aligned
    TODO: change to the loop instead of the segments ?
    """

    def __init__(self, event="boundaries"):
        raise DeprecationWarning()
        self.event = event
        super(EventsAlignmentRule, self).__init__()

    def getEvent(self, track):
        if self.event == "boundaries":
            return track.features["boundaries"]
        if self.event == "kick":
            return [
                float(onset[0]) for onset in track.adtOnsets if onset[1] == "0"
            ]
        if self.event == "beat":
            return track.getBeats()

    def run(self, mix, boundaries):
        tracks = Rule.getTracks(mix, boundaries)
        # set threshold of deviations
        # 20Hz, lowest hearable frequency. below 50ms between two notes, we (should) hear only one note
        # if the deviation is above an eighth note it's a different beat, thus it's ok.
        minThreshold = 0.05
        # deckTempo = max([track.features["tempo"] * track.playRate for track in tracks])

        # compute the deck's location of each event
        # we also remove events outside of the overlaping areas
        # We still need to align before and after the boundaries of each tracks because we feel the structure/beat in long period of time

        # returns: beforeOverlapA, startOverlapA, endTrackA, afterEndTrackA, startTrackB, endOverlapB, afterOverlapB
        # localTimes = [
        #     Rule.getTransitionLocalTimes(
        #         tracks[i], tracks[i + 1], windowInBeats=window)
        #     for i in range(len(tracks) - 1)
        # ]
        localTimes = Rule.getTransitionsLocalTimes(tracks)

        overlapsEvents = [([
            tracks[i].getDeckTime(event, enableExceeding=False)
            for event in self.getEvent(tracks[i])
            if event > localTimes[i][0] and event < localTimes[i][3]
        ], [
            tracks[i + 1].getDeckTime(event, enableExceeding=False)
            for event in self.getEvent(tracks[i + 1])
            if event > localTimes[i][4] and event < localTimes[i][6]
        ]) for i in range(len(tracks) - 1)]

        # compute the distance between events for each overlaps
        overlapErrors = [
            np.abs(quantization.diff(trackAEvents,
                                     trackBEvents, maxThreshold=10000))
            for trackAEvents, trackBEvents in overlapsEvents
        ]

        # if no segments can be aligned for one transition, it's 0
        if not len([overlap for overlap in overlapErrors if len(overlap)]):
            return 0

        # add a 1 or a 0 for each event which should overlap (distance < maxDistance) if the difference is perceptible (distance > minDistance)
        result = np.min([
            np.mean([
                1 if distance < minThreshold else 0 for distance in distances
            ]) for distances in overlapErrors
        ])

        return result

    def __str__(self):
        return self.event + "overlap"
