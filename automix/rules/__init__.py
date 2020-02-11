# from https://www.youtube.com/watch?v=RvvitRlYClU
# Loops of (4-8-16) beats should be always in sync. If the bass line of the first track stop, the second one should begin
# if there is two bass line, remove one with the EQ
# if the loop is 4 beats long. Just repeat it 4 times.
# it's ok if two tracks are not in phase as long as they share the same tempo

# From Paolo:
# If there is a lot going on in the music (i.e. anything which is not kicks) we should not fade it slowly but

# From me:
# We should not overlap a ternary piece with a binary one
# we should compute the correlation of the overlaping. the better the correlation, the better the transition

# From https://www.scientificamerican.com/article/psychology-workout-music/
# cadence in the lyrics is very important to give the energy of a piece opf music.
# tempo are: 120bpm is the most common. 160-180bpm if running on a treadmile. 145bpm is the ceiling though

# From https://www.pyramind.com/training/online/dj101-will-marshall-3-6
# EQ
# try not to add a boost if you can avoid it because it distord the audio (but you could remove the other bands and turn up the gain)
# you can slowly incorporate the highs, but dependingon the contest, you may not have to
# you want to swap the basse
# The style of the EQ depends on the style of the track

import numpy as np
import sys

from automix.rules.harmonicTransitionRule import HarmonicTransitionRule
from automix.rules.percussiveTransitionRule import PercussiveTransitionRule
from automix.rules.activityRule import ActivityRule
from automix.rules.eventsAlignmentRule import EventsAlignmentRule
from automix.rules.suitableKeyRule import SuitableKeyRule
from automix.rules.veireTransitionsRule import VeireTransitionRule

def runAll(mix, boundaries, switches):
    """
    run all the rules for this mix.
    Return the weighted average score as well as a string describing the individual scores for debuging

    Parameters:
        mix: an array of Deck containing tracks as weel as effect
        boundaries: a tuple indicating the boundaries in seconds where the rules should be applied. if None, the whole mix is going to be analyzed
    """
    if boundaries is None:
        boundaries = [0, sys.maxsize]  # TODO: lookup the real boundaries

    rules = [
        VeireTransitionRule()
        # Veire: transitions types (relaxed, rolling, double drop), vocal clash detection, onset detection matching
        # EventsAlignmentRule(), ActivityRule(weight=2)
    ]  # MaxPlayrateChangeRule(),  BeatOverlapPrecisionRule(), SuitableKeyRule(), HarmonicTransitionRule(), PercussiveTransitionRule()
    average = 0.
    ruleComputed = 0
    logs = []
    for rule in rules:
        result = rule.run(mix, boundaries, switches)
        logs.append((str(rule), result))  # logging.info(str(rule), result)
        if not np.isnan(result):
            average += result * rule.weight
            ruleComputed += rule.weight
    if ruleComputed:
        return average / ruleComputed, logs
    else:
        return 0, logs
