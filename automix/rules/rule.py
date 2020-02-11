from typing import List
from automix.model.classes.track import Track

class Rule(object):
    """
    abstract class which holds a rule
    """

    def __init__(self, weight=1):
        self.weight = weight
        self.description = self.__class__.__name__

    def __str__(self):
        return self.description

    def run(self, mix, **kwarg):
        """
        return a value between 0 and 1 telling how well the rule is applied in the transition

        Args:
            mix (Deck[]): all the information from the data model.
                A mix is a collection of Decks containing Tracks at specific positions and effects
            kwarg:
                - Boundaries
                - switches
        """
        raise NotImplementedError("To be implemented")

    @staticmethod
    def getTransitionLocalTimes(trackA, trackB, windowInBeats=16):
        """
        returns Specific times from the overlapings between two tracks
        the window is time before and after the POIs
        """
        windowInSeconds = windowInBeats * 60 / \
            trackA.features["tempo"] / trackA.playRate
        # times in the track timeline
        beforeOverlapA = trackA.getTrackTime(trackB.position - windowInSeconds)
        startOverlapA = trackA.getTrackTime(trackB.position)
        endTrackA = trackA.getTrackTime(trackA.length + trackA.position)
        afterEndTrackA = trackA.getTrackTime(trackA.length + trackA.position +
                                             windowInSeconds)
        startTrackB = trackB.soffs
        endOverlapB = trackB.getTrackTime(trackA.getDeckTime(endTrackA))
        afterOverlapB = trackB.getTrackTime(
            trackA.getDeckTime(endTrackA) + windowInSeconds)

        return beforeOverlapA, startOverlapA, endTrackA, afterEndTrackA, startTrackB, endOverlapB, afterOverlapB

    @staticmethod
    def getTransitionsLocalTimes(tracks):
        """
        sort the tracks and returns Specific times from the overlaps in the tracks provided
        """
        # tracks = Rule.getTracks(mix, boundaries)
        tracks.sort(key=lambda track: track.position)
        return [Rule.getTransitionLocalTimes(tracks[i], tracks[i+1]) for i in range(0, len(tracks)-1, 2)]

    @staticmethod
    def getTracks(mix, boundaries) -> List[Track]:
        """
        return the unordered tracks from the mix
        """
        return [track for deck in mix for track in deck.tracks if boundaries[0] < track.getEndPosition() and boundaries[1] > track.position]
