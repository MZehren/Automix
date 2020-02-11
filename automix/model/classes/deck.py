"""
Deck container
"""
import copy
from typing import List

from automix.model.classes.track import Track


class Deck(object):
    """
        Data structure to contain a Deck with effects. 
        It corresponds to tracks in Reaper 
    """

    def __init__(self, name: str = "", tracks: List[Track] = None):
        self.tracks = tracks if tracks else []
        self.name = name

        self.FX = {"gainPt": [], "volPt": [], "lowPt": [], "midPt": [], "highPt": [], "hpfPt": [], "lpfPt": []}

    def append(self, track: Track):
        """
        Add a track to the deck
        """
        self.tracks.append(track)

    def updateFX(self):
        """
        Apply the track effects to the Deck
        """
        for track in self.tracks:
            for FX, values in track.FX.items():
                if FX not in self.FX:
                    self.FX[FX] = []
                for pt in values:
                    newPt = copy.copy(pt)
                    newPt.position = track.getDeckTime(pt.position)
                    self.FX[FX].append(newPt)
