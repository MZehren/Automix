from numpy import mean

from automix.featureExtraction.key import edmkeyProxy as edm
from automix.rules.transitionRule import TransitionRule


class SuitableKeyRule(TransitionRule):
    """
    Rule rating the transition between two tracks based on how well their keys
    fit according to the mirex score.
    """

    @staticmethod
    def score(trackA, trackB):
        convertedKeyA = edm.convertKey(trackA.getKey())
        convertedKeyB = edm.convertKey(trackB.getKey())
        return edm.mirexScore(convertedKeyA, convertedKeyB)