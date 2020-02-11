from automix.rules.rule import Rule
from automix.rules.transitionRule import TransitionRule


class HarmonicTransitionRule(TransitionRule):
    """
    Rule rating the transition between two tracks high if and only if at maximum
    one of the transitioning parts is harmonic.
    """

    @staticmethod
    def score(trackA, trackB):
        _, startOverlapA, endTrackA, _, startTrackB, endOverlapB, _ \
                    = Rule.getTransitionLocalTimes(trackA, trackB)

        def hasHarmonicTransition(track, start, end):
            values = track.getHarmonic().getValues(start, end)
            if not values:
                return False
            return max(values)

        return float(not hasHarmonicTransition(trackA, startOverlapA, endTrackA)
                     or not hasHarmonicTransition(trackB, startTrackB, endOverlapB))
