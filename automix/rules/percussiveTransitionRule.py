from automix.rules.rule import Rule
from automix.rules.transitionRule import TransitionRule


class PercussiveTransitionRule(TransitionRule):
    """
    Rule rating the transition between two tracks high if and only if at maximum
    one of the transitioning parts is purely percussive.
    """

    @staticmethod
    def score(trackA, trackB):
        _, startOverlapA, endTrackA, _, startTrackB, endOverlapB, _ \
                    = Rule.getTransitionLocalTimes(trackA, trackB)

        def hasPurelyPercussiveTransition(track, start, end):
            harmonic = track.getHarmonic().getValues(start, end)
            if not harmonic or max(harmonic):
                return False
            return max(track.getPercussive().getValues(start, end))

        return float(
            not hasPurelyPercussiveTransition(trackA, startOverlapA, endTrackA)
            or not hasPurelyPercussiveTransition(trackB, startTrackB,
                                                 endOverlapB))
