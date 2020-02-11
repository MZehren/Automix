import json


class FeatureSerializer(object):
    """
    Serialize and deserialize features from a path
    """

    @staticmethod
    def deserialize(path):
        """ 
        Returns a dictionary containing the json content of the file
        """
        serializedFeatures = {}
        try:
            with open(path) as file:
                serializedFeatures = decode(file.read())
        except Exception:
            return None

        return serializedFeatures

    @staticmethod
    def serialize(path, features):
        """
        Writte the features in json
        Features has to be/contain a dict, list, primitive, or object implementing jsonEncode(void):any
        """
        with open(path, 'w') as featuresFile:
            featuresFile.write(json.dumps(features, cls=MyJSONEncoder))


class MyJSONEncoder(json.JSONEncoder):
    """
    My own version of JSONEncoders
    This implementation takes care of objects containing a jsonEncode method
    """

    def default(self, obj):  # pylint: disable=E0202
        try:
            result = obj.jsonEncode()  #TODO: make it explicit that the objects need to implement jsonEncode
            if isinstance(result, dict):
                result["type"] = str(type(obj))
            return result
        except Exception:
            pass

        try:
            return obj.tolist()
        except Exception:
            pass

        return json.JSONEncoder.default(self, obj)


def decode(stringObject):
    jsonObject = json.loads(stringObject)
    return recursiveMap(jsonObject)


def recursiveMap(obj):
    """
    recursively map all the fields of the json decoded object to class from the model
    """
    try:
        from automix.model.classes.signal import Signal, SparseSignal, SparseSegmentSignal
        if isinstance(obj, dict):
            if u'type' in obj and (obj[u"type"] == str(Signal) or obj[u"type"] == str(SparseSignal)):
                obj = Signal.jsonDeserialize(obj)
            elif u'type' in obj and obj[u"type"] == str(SparseSegmentSignal):
                obj = SparseSegmentSignal.jsonDeserialize(obj)
            else:
                for key, value in obj.items():
                    obj[key] = recursiveMap(obj[key])
        elif isinstance(obj, list):
            for key, value in enumerate(obj):
                obj[key] = recursiveMap(obj[key])
    except Exception:
        pass

    return obj
