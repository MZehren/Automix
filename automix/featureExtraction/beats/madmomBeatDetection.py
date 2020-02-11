"""
container for the downbeat estimator implemented with  
"""
# encoding: utf-8
import madmom
import numpy as np

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.utils import quantization
from automix.model.classes.signal import Signal
import logging


class MadmomBeatDetection(Estimator):
    """
    Compute the beat, downbeat, tempo of a track

    Parameters:
    -    path (string): location of the track
    -    parameterSnapDistance=0.05: Distance threshold at which the actual beat location is discarded and the
        expected beat location from the tempo is used.
    -   parameterTransitionLambda=300, default to 100 in madmom. Controls the probability that the DBN hidden state transition to another tempo
    -   parameterCorrectToActivation=True
    """

    def __init__(
            self,
            parameterSnapDistance=0.05,
            parameterTransitionLambda=300, 
            parameterCorrectToActivation=True,
            inputPath="path",
            outputBeats="beats",
            outputdownbeats="downbeats",
            outputStrongBeats="strongBeats",
            outputTempo="tempo",
            cachingLevel=0,
            forceRefreshCache=False):
        self.parameters = {
            "snapDistance": Parameter(parameterSnapDistance),
            "transitionLambda": Parameter(parameterTransitionLambda),
            "correctToActivation": Parameter(parameterCorrectToActivation)
        }
        self.inputs = [inputPath]
        self.outputs = [outputBeats, outputdownbeats, outputStrongBeats, outputTempo]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def plot(self, beats):
        import matplotlib.pyplot as plt
        times = beats[:, 0]
        plt.plot(np.diff(times))
        plt.show()

    def predictOne(self, path):

        # call madmom to get beats
        fps = 100
        act = madmom.features.RNNDownBeatProcessor()(str(path))
        proc = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4],
                                                            fps=fps,
                                                            transition_lambda=self.parameters["transitionLambda"].value,
                                                            correct=self.parameters["correctToActivation"].value)
        beats = proc(act)
        if len([beat for i, beat in enumerate(beats) if (i + beats[0][1] - 1) % 4 + 1 != beat[1]]):
            logging.error("Beat detection skipped a beat")
        # get the tempo
        # evenGrids = quantization.separateInEvenGrids(beats[:, 0], regularityThreshold=self.parameters["snapDistance"].value)
        # longuestEvenGridIndex = np.argmax([len(grid) for grid in evenGrids])
        # tau = np.average([(evenGrid[-1] - evenGrid[0]) / (len(evenGrid) - 1) for evenGrid in evenGrids if len(evenGrid) > 1],
        #                  weights=[len(evenGrid)
        #                           for evenGrid in evenGrids if len(evenGrid) > 1]) * fps  # TODO: use only the longest portion ?
        # tempo = 60 * fps / tau
        # beatLength = tau / fps  # i.e 0.5s
        # refBeat = [beat for beat in beats if beat[0] == evenGrids[longuestEvenGridIndex][0]][0]

        # # extend the grid of beats to remove holes in it
        # trackLength = float(len(act)) / fps
        # extendedBeats = quantization.extendGrid(refBeat,
        #                                         beats,
        #                                         trackLength,
        #                                         beatLength,
        #                                         SnapDistance=self.parameters["snapDistance"].value)
        tempo = 60 / np.mean(np.diff(np.array(beats)[:, 0]))

        # Get the confidence as the mean of the activation at each GT beat. Sums the two outputs of the NN
        # beat = self._getConfidence(act, beat, fps, extendedBeats)
        beatsT = [beat[0] for beat in beats]
        downbeatsT = [beat[0] for beat in beats if beat[1] == 1]
        strongBeatsT = [beat[0] for beat in beats if beat[1] == 1 or beat[1] == 3]
        return (Signal(np.ones(len(beatsT)), times=beatsT,
                       sparse=True), Signal(np.ones(len(downbeatsT)), times=downbeatsT,
                                            sparse=True), Signal(np.ones(len(strongBeatsT)), times=strongBeatsT,
                                                                 sparse=True), tempo)

    def _getConfidence(self, act, fps, extendedBeats):
        # Get the confidence as the mean of the activation at each GT beat. Sums the two outputs of the NN
        activationPerBeat = [np.sum(act[int(beat * fps)]) for beat in np.array(extendedBeats)[:, 0]]

        beatConfidence = np.mean(activationPerBeat)

        # get the confidence between beats
        beatSamples = [int(beat * fps) for beat in np.array(extendedBeats)[:, 0]]
        beatSamples = np.append([[sample - 1, sample, sample + 1] for sample in beatSamples], [])
        activationBetweenBeats = [np.sum(act[i]) for i in range(len(act)) if i in beatSamples]
        interBeatConfidence = np.mean([np.sum(act[i]) for i in range(len(act)) if i not in beatSamples])
        return beatConfidence, interBeatConfidence, activationBetweenBeats
