"""
container for the vocal separation estimator
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pkg_resources import resource_filename
import importlib

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class VocalMelodyExtraction(Estimator):
    """
    estimator based on https://github.com/s603122001/Vocal-Melody-Extraction
    """

    def __init__(self,
                 inputPath="path",
                 outputClassification="vocals",
                 outputPitch="vocalsMelody",
                 parameterModel="Seg",
                 cachingLevel=0,
                 forceRefreshCache=False):
        self.inputs = [inputPath]
        self.outputs = [outputClassification, outputPitch]
        self.parameters = {"model": Parameter(parameterModel)}
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, path: str):
        """
        method copied from the main file in the project
        """
        # pkg_resources.()
        # project = importlib.import_module("vendors.Vocal-Melody-Extraction.project")
        from project.MelodyExt import feature_extraction
        from project.utils import load_model, save_model, matrix_parser
        from project.test import inference
        from project.model import seg, seg_pnn, sparse_loss
        from project.train import train_audio

        # load wav
        song = path

        # Feature extraction
        feature = feature_extraction(song)
        feature = np.transpose(feature[0:4], axes=(2, 1, 0))

        # load model

        model = load_model(
            resource_filename(__name__,
                              "../../../vendors/Vocal-Melody-Extraction/Pretrained_models/" + self.parameters["model"].value))
        batch_size_test = 10
        # Inference
        print(feature[:, :, 0].shape)
        extract_result = inference(feature=feature[:, :, 0], model=model, batch_size=batch_size_test)

        # Output
        r = matrix_parser(extract_result)
        return (Signal(r[:, 0], sampleRate=50), Signal(r[:, 1], sampleRate=50))
