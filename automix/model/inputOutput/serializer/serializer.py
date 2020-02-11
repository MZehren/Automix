from typing import List

class Serializer(object):
    """
    Abstract class for serializers
    """

    # def serializeTrack(self, path, track: Track):
    #     """
    #     Base method to serialize a mix
    #     """
    #     raise NotImplementedError()

    # def serializeMix(self, path: str, decks: List[Deck], BPM=120, markers: List = [], subfolder: str = ""):
    #     """
    #     Base method to serialize a mix
    #     """
    #     raise NotImplementedError()

    # @staticmethod
    # def deserializeTrack(path: str) -> Track:
    #     """
    #     Base static method to deserialize a track "settings"
    #     """
    #     raise NotImplementedError()

    # @staticmethod
    # def loadFolder(path: str):
    #     """
    #     Base method to deserialize a folder
    #     """
    #     result = []
    #     for root, dirs, files in os.walk(path):
    #         for file in files:
    #             result.append(deserializeTrack)
