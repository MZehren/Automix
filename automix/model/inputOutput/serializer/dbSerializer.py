import mongoengine
from pymongo import MongoClient

from automix import config
from automix.model.inputOutput.serializer.serializer import Serializer


class DBSerializer(Serializer):
    """
    class handling jams serialization of track or mixes
    """

    def __init__(self):
        client = MongoClient()
        self.db = client["1001tracklists"]
        self.mixes = self.db.mixes
        # TODO singleton?

    def insert(self, mix, id):
        mix["_id"] = id
        self.mixes.insert_one(mix)

    def retrieive(self, match):
        return self.mixes.find(match)

    def delete(self, match):
        self.mixes.remove(match)

    def exist(self, match):
        result = self.retrieive(match)
        return result.count()



