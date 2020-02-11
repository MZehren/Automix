import xml.etree.ElementTree as ET

from automix.model.classes import Signal
from automix.model.classes.track import Track

class TraktorSerializer(object):
    @staticmethod
    def tracktorDeserialize(path, titles=None):
        """
        get a track from the xml format from tracktor (1?)
        """
        tree = ET.parse(path)
        root = tree.getroot()
        tracks = {}
        for entry in root.find("COLLECTION").iter("ENTRY"):
            track = Track()
            track.name = entry.attrib["TITLE"]
            track.path = entry.find("LOCATION").attrib["FILE"][:-4] #Removing .mp3
            cues = [cue for cue in entry.iter("CUE_V2") if cue.attrib["NAME"] != "AutoGrid"]
            track.features["Cues"] = Signal([cue.attrib["NAME"][:7] for cue in cues],
                                             times=[float(cue.attrib["START"]) / 1000 for cue in cues],
                                             sparse=True)
            tracks[track.path] = track
        if titles: 
            return [tracks[t] if t in tracks else None for t in titles]
        return tracks.values()


# bla = TraktorSerializer.tracktorDeserialize(
#     "/home/mickael/Documents/programming/dj-tracks-switch-points/evaluation/mixed in key/collection.nml")
# print(bla)