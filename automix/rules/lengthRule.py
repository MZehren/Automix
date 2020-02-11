import numpy as np

from backend.rules.rule import Rule


class LengthRule(Rule):
    """
    The mixed track has to be played for it's majority otherwise it will not be good
    """

    def run(self, mix):
        tracks = Rule.getTracks(mix)
        return np.mean([
            track.length / (track.duration / track.playRate)
            for track in tracks
        ])

    def __str__(self):
        return "Majority played"
