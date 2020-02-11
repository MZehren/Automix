"""
Proxy to the music structure analysis framework
"""

import numpy as np

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import SparseSignal


class MsafProxy(Estimator):
    """
    run MSAF: https://github.com/urinieto/msaf

    * inputPath: input feature containing the path to the audio file
    * outputSignal: name of the output feature
    * algorithm: name of the algorithm 
    * feature: name of the feature to use
    presented combo in the article are : ["scluster", None] ["olda", None], ["sf", "cqt"]
    or 
    feat_dict = {
        'sf': 'pcp',
        'levy': 'pcp',
        'foote': 'pcp',
        'siplca': '',
        'olda': '',
        'cnmf': 'pcp',
        '2dfmc': ''
    }
    """

    def __init__(self, inputPath="path", outputSignal="msaf-scluster", algorithm="scluster", feature=None):
        super().__init__(parameters={"algorithm": algorithm, "feature": feature}, inputs=[inputPath], outputs=[outputSignal])

    def predictOne(self, path):
        """
        Returns the structure and label from the algorithm specified
        Removes the first and last boundarie which is the start and the end of the track
        """
        import msaf
        if self.parameters["feature"].value is None:
            boundaries, labels = msaf.process(path, boundaries_id=self.parameters["algorithm"].value)
        else:
            boundaries, labels = msaf.process(path,
                                              boundaries_id=self.parameters["algorithm"].value,
                                              feature=self.parameters["feature"].value)
        return (SparseSignal(labels[1:], boundaries[1:-1]), )
