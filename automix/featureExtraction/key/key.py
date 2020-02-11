from automix.featureExtraction import Estimator, Parameter
import madmom


class Key(Estimator):
    """
    Estimator calculating the key of a given audio.
    """

    def __init__(self, input="path", output="key", cachingLevel=0, forceRefreshCache=False):
        self.inputs = [input]
        self.outputs = [output]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache
        self.parameters = {}

    def predictOne(self, path):
        """Estimates the key of a given audio file.

        Args:
            path (str): The absolute path of the audio file.
    
        Returns:
            tuple of str: The key of the audio file.

        """
        return self.madmomKey(path)

    def madmomKey(self, file):
        proc = madmom.features.key.CNNKeyRecognitionProcessor()
        proba = proc(file)
        key = madmom.features.key.key_prediction_to_label(proba)
        return (key, )

    def edmKey(self):
        """
        Deprecated
        """
        raise DeprecationWarning()
        from featureExtraction.key import edmkeyProxy as edm
        self.loadParams()
        edm.setParameters(self.parameters)
        key = edm.estimateKey(path, self.getValue("separator"))
        return (key, )