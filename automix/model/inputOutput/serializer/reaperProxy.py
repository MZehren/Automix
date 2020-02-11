"""
generates a reaper project file
"""
import re
from enum import Enum
from typing import List, Tuple

import librosa
import numpy as np
import pkg_resources

from automix.model.classes.deck import Deck
from automix.model.classes.point import Point
from automix.model.inputOutput.serializer.serializer import Serializer


class ReaperProxy(Serializer):
    """
    Serialize a mix in Reaper's .RPP format
    """
    # Reaper project template files
    PROJECTTEMPLATE = pkg_resources.resource_string(__name__, "../template/project.template.RPP").decode()
    TRACKTEMPLATE = pkg_resources.resource_string(__name__, "../template/track.template.RPP").decode()
    ITEMTEMPLATE = pkg_resources.resource_string(__name__, "../template/item.template.RPP").decode()

    @staticmethod
    def getMarker(time=0, label="x", color="r"):
        """
        returns a marker for Reaper
        """
        colorDict = {"r": "0", "b": "33521664"}
        return str(time) + " " + str(label).replace(" ", "_") + " 0 " + colorDict[color] + " 1 R"

    def __init__(self):
        self.globalId = 0

    def serialize(self,
                  path: str,
                  decks: List[Deck],
                  BPM: float = 120,
                  markers: List[str] = [],
                  tempoMarkers: List[str] = [],
                  subFolder: str = ""):
        """
        Call getReaperProject and write the file to disk 
        """
        with open(path, 'w') as outputFile:
            outputFile.write(
                self.getReaperProject(BPM=BPM, decks=decks, markers=markers, tempoMarkers=tempoMarkers, subFolder=subFolder))

    def __getUniqueID(self):
        """
        Returns a unique ID based on a global incremented value
        """
        self.globalId += 1
        return self.globalId

    def __clean(self, string):  # TODO: is this the good location to put this function ?
        """
        Cleans a string by removing all the special characters
        """
        return re.sub(r'[^\x00-\x7f]', r'', string)

    def __getPTString(self, position, amplitude, shape, curve):
        """
        Returns the string of an effect envelope point
        Shape is: TODO 
        """
        return "PT " + str(position) + " " + str(amplitude) + " " + str(shape.value) + " 0 0 0 " + str(curve)

    def __getSMString(self, tuplePositions: List[Tuple[float, float]], progressive=False):
        """
        Returns the string of a stretch marker
        - the format is: 
            - SM PositionInRealTime1 trackPositionInTrackTime1 slope1 + PositionInRealTime2 trackPositionInTrackTime2 slope2
        - So if you want to play the track such as the first second in track takes two second to play:
            - "SM 0 0 0 + 2 1 0" 
        - Slope dictates what is the starting and end ratio of the segment with the formula
            - meanPR = TrackSegmentDuration / realTimeDuration 
            - startSpeed = MeanPR-(MeanPR * slope)
            - endSpeed = MeanPR+(MeanPR * slope)
        - or 
            - Slope = EndPR/MeanPR - 1
            - Slope = 1 -StartPR/MeanPR
        - With previous example, the Mean playback rate is MeanPR = 1/2 = 0.5. If you want the end speed to be 1:
            - Slope = 1/0.5 - 1 = 1
            - You need to writte "SM 0 0 1 + 2 1 0" 
        """
        slopes = np.zeros(len(tuplePositions))
        if progressive:
            pass
        return "SM " + " + ".join([str(p[0]) + " " + str(p[1]) + " " + str(slopes[i]) for i, p in enumerate(tuplePositions)])

    def getReaperProject(self, BPM=120, markers=[], tempoMarkers=[], decks=[], subFolder=""):
        """
            generate text of a reaper project with
            tempo:int,
            markers:see reaperProxy.getMarker,
            decks:Deck,
            and a subfolder:text. Path prepended to the track's path when relative path are used
        """
        self.globalId = 0
        return self.PROJECTTEMPLATE \
            .replace("[BPM]", str(BPM)) \
            .replace("[TEMPOENVEX]", "\n".join(["PT " + str(temps)  + " " + str(tempo) + " 1" for temps, tempo in tempoMarkers])) \
            .replace("[MARKERS]", "\n".join(["MARKER " + str(i) + " " + str(marker) for i, marker in enumerate(markers)])) \
            .replace("[TRACKS]", "\n".join([self.__getReaperTrack(deck, subFolder) for deck in decks]))

    def __getReaperTrack(self, deck, subFolder):
        """
        Generate the text for a reaper track, AKA a DJ deck. this part contains all the effects applied to the mix

        The values are
        PT      122.85883415     1                      5       0     (1)                 (0) (0.9)
        PT      position(second) value(arbitrary unit)  shape   ?     selection(boolean)  ?   curve(-1 = fast start, 0 = linear, 1 = fast end)

        Same for the faders
        FADEOUT 1 33.644352 0 1 0 0.25

        The shapes are:
        0: linear, 1: square, 2: slow start/end, 3: fast start, 4: fast end, 5: bezier

        """
        return self.TRACKTEMPLATE \
            .replace("[TRACKID]", str(self.__getUniqueID())) \
            .replace("[TRACKNAME]", str(deck.name)) \
            .replace("[GAINVIS]", "1" if deck.FX["gainPt"] else "0") \
            .replace("[GAINPT]", "\n".join([self.__getPTString(pt.position, librosa.core.db_to_amplitude(pt.amplitude, ref=1), pt.shape, pt.curve) for pt in deck.FX["gainPt"]])) \
            .replace("[VOLVIS]", "1" if deck.FX["volPt"] else "0") \
            .replace("[VOLPT]", "\n".join([self.__getPTString(pt.position, librosa.core.db_to_amplitude(pt.amplitude, ref=1), pt.shape, pt.curve) for pt in deck.FX["volPt"]])) \
            .replace("[3BandEQID]", str(self.__getUniqueID())) \
            .replace("[LOWVIS]", "1" if deck.FX["lowPt"] else "0") \
            .replace("[LOWPT]", "\n".join([self.__getPTString(pt.position, max(librosa.core.db_to_amplitude(pt.amplitude, ref=0.25), 0.005), pt.shape, pt.curve) for pt in deck.FX["lowPt"]])) \
            .replace("[MIDVIS]", "1" if deck.FX["midPt"] else "0") \
            .replace("[MIDPT]", "\n".join([self.__getPTString(pt.position, max(librosa.core.db_to_amplitude(pt.amplitude, ref=0.25), 0.005), pt.shape, pt.curve) for pt in deck.FX["midPt"]])) \
            .replace("[HIGHVIS]", "1" if deck.FX["highPt"] else "0") \
            .replace("[HIGHPT]", "\n".join([self.__getPTString(pt.position, max(librosa.core.db_to_amplitude(pt.amplitude, ref=0.25), 0.005), pt.shape, pt.curve) for pt in deck.FX["highPt"]])) \
            .replace("[HPFLPFID]", str(self.__getUniqueID())) \
            .replace("[HPFVIS]", "1" if deck.FX["hpfPt"] else "0") \
            .replace("[HPFPT]", "\n".join([self.__getPTString(pt.position, max(librosa.core.db_to_amplitude(pt.amplitude, ref=0.25), 0.005), pt.shape, pt.curve) for pt in deck.FX["hpfPt"]])) \
            .replace("[LPFVIS]", "1" if deck.FX["lpfPt"] else "0") \
            .replace("[LPFPT]", "\n".join([self.__getPTString(pt.position, max(librosa.core.db_to_amplitude(pt.amplitude, ref=0.25), 0.005), pt.shape, pt.curve) for pt in deck.FX["lpfPt"]])) \
            .replace("[ITEMS]", "\n".join([self.__getReaperItem(track, subFolder) for track in deck.tracks])) \

    def __getReaperItem(self, track, subFolder):
        """
        Generates the text of an item AKA a DJ track
        """
        return self.ITEMTEMPLATE \
            .replace("[ITEMNAME]", self.__clean(track.name)) \
            .replace("[ITEMPOSITION]", str(track.position)) \
            .replace("[ITEMLENGTH]", str(track.length) if hasattr(track, "length") else str(track.duration)) \
            .replace("[ITEMIGUID]", str(self.__getUniqueID())) \
            .replace("[ITEMSOFFS]", str(track.soffs)) \
            .replace("[ITEMPLAYRATE]", str(track.playRate)) \
            .replace("[ITEMPRESERVEPITCH]", str(track.preservePitch)) \
            .replace("[ITEMGUID]", str(self.__getUniqueID())) \
            .replace("[FADEIN]", str(track.fadeIn)) \
            .replace("[FADEOUT]", str(track.fadeOut)) \
            .replace("[SOURCETYPE]", track.sourceType) \
            .replace("[ITEMFILE]", subFolder + self.__clean(track.path)) \
            .replace("[STRETCHMARKER]", self.__getSMString(track.stretchMarkers, progressive=track.stretchMarkersProgressive))
