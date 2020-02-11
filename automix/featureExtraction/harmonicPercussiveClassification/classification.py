import copy
import itertools
import librosa as lr
import numpy as np

from featureExtraction.estimator import Estimator
from model.classes.signal import Signal
from utils.perceptiveLoudness import loudness as ld


class Classification(Estimator):
    """
    Estimator determining for each bar of a track whether that bar is harmonic
    and whether it is percussive.
    """

    def __init__(self):
        self.loadParams()
        self.inputs = ["barSamples", "sampleRate", "loudness"]
        self.outputs = ["harmonic", "percussive"]
        self.cachingLevel = 0

    def _calcNumFFTBins(self, sampleRate):
        """Scales the parameter numFFTBinsAt44100Hz according to the sampling
        frequency.

        Args:
            sampleRate (int): The sampling frequency.

        Returns:
            int: The scaled number of FFT bins.

        """

        # numFFTBins should be proportional to sampleRate
        ratioFreq = sampleRate / 44100.0
        numFFTBins = self.getValue("numFFTBinsAt44100Hz") * ratioFreq
        return int(round(numFFTBins * 0.5)) * 2

    def _separate(self, y, numFFTBins):
        """Decomposes a signal into two signals, one containing only the
        harmonic parts and the other containing only the percussive parts.

        Args:
            y (list of float): The signal.
            numFFTBins (int): The number of FFT bins.

        Returns:
            list of float: The harmonic signal.
            list of float: The percussive signal.

        """

        # Convert signal to spectrum
        spectrum = lr.stft(y, n_fft=numFFTBins)

        # Decompose into harmonic/percussive
        spectrumHarmonic, spectrumPercussive = lr.decompose.hpss(
            spectrum,
            kernel_size=(self.getValue("medianWidthHarmonic"),
                         self.getValue("medianWidthPercussive")),
            margin=(self.getValue("marginHarmonic"),
                    self.getValue("marginPercussive")))

        # Convert back to signals
        yHarmonic = lr.istft(spectrumHarmonic)
        yPercussive = lr.istft(spectrumPercussive)
        return yHarmonic, yPercussive

    @staticmethod
    def _loudness(part, sampleRate, loudness):
        """Calculates the absolute loudness of the harmonic or percussive part
        of a song.

        Args:
            part (list of float): The harmonic or percussive part as samples.
            sampleRate (int): The sample rate of the song in Hz.
            loudness (float): The loudness of the song in dB.

        Returns:
            float: The absolute loudness of the part.

        """

        return ld.loudnessSignal(part, sampleRate) - loudness

    def _calcLoudnesses(self, barSamples, sampleRate, loudness):
        """Calculates the absolute loudnesses of the harmonic and percussive
        parts for each bar of a song.

        Args:
            barSamples (list of list of float): The samples of each bar.
            sampleRate (int): The sample rate of the song in Hz.
            loudness (float): The loudness of the song in dB.

        Returns:
            list of tuple of float: The absolute loudnesses for each part for
            each bar.

        """

        numFFTBins = self._calcNumFFTBins(sampleRate)
        loudnesses = []
        for yBar in barSamples:
            yHarmonic, yPercussive = self._separate(yBar, numFFTBins)
            loudnessHarmonic = self._loudness(yHarmonic, sampleRate, loudness)
            loudnessPercussive = self._loudness(yPercussive, sampleRate,
                                                loudness)
            loudnesses.append((loudnessHarmonic, loudnessPercussive))
        return loudnesses

    def predictOne(self, barSamples, sampleRate, loudness):
        """Classifies each bar of a track as either harmonic or non-harmonic and
        as either percussive or non-percussive

        Args:
            barSamples (Signal): The samples for each bar.
            loudness (float): The loudness of the song considered.
            sampleRate (float): The sample rate of the song considered.

        Returns:
            tuple of Signal: The signal containing the harmonic classification
                result for each bar, and the signal containing the percussive
                classification for each bar.

        """

        # Obtain loudness of each bar and part
        loudnesses = self._calcLoudnesses(barSamples, sampleRate, loudness)

        # Classify by thresholds
        classificationsHarmonic = []
        classificationsPercussive = []
        for loudnessHarmonic, loudnessPercussive in loudnesses:
            isHarmonic = loudnessHarmonic > self.getValue("thresholdHarmonic")
            classificationsHarmonic.append(isHarmonic)

            isPercussive = loudnessPercussive > self.getValue(
                "thresholdPercussive")
            classificationsPercussive.append(isPercussive)

        # Wrap result lists into signals
        signalHarmonic = Signal(classificationsHarmonic, barSamples.times)
        signalPercussive = Signal(classificationsPercussive, barSamples.times)
        return (signalHarmonic, signalPercussive)

    @staticmethod
    def _score(metric,
               estimations,
               annotations,
               loudnessesBelow,
               loudnessesAbove):
        """Returns a score between 0 and 1 grading the performance of either the
        harmonic or percussive classification.

        Args:
            metric (str): The metric which is applied.
            estimations (list of bool): The estimated labels.
            annotations (list of bool): The annotated labels.
            loudnessesBelow (list of float): The calculated loudnesses below the
                classifying threshold.
            loudnessesAbove (list of float): The calculated loudnesses above the
                classifying threshold.

        Returns:
            float: The score.

        """

        def accuracy():
            return np.equal(estimations, annotations).mean()
        
        def falsePositiveRate():
            return np.logical_and(
                estimations, np.logical_not(annotations)).mean()

        def falseNegativeRate():
            return np.logical_and(
                np.logical_not(estimations), annotations).mean()

        def weightedAccuracy():
            # accuracy with greater weight on rarer label
            
            numClassifications = len(annotations)
            numPositives = np.sum(annotations)
            numNegatives = numClassifications - numPositives

            if numPositives > 0 and numNegatives > 0:
                scoreVector = np.full(numClassifications, 0.5 / numPositives) \
                            * annotations \
                            + np.full(numClassifications, 0.5 / numNegatives) \
                            * np.logical_not(annotations)
                return np.sum(np.equal(estimations, annotations) * scoreVector)
            
            return accuracy()
        
        if metric == "accuracy":
            return accuracy()
        if metric == "falsePositiveRate":
            return 1 - falsePositiveRate()
        if metric == "falseNegativeRate":
            return 1 - falseNegativeRate()
        if metric == "weightedAccuracy":
            return weightedAccuracy()
        raise RuntimeError("Unknown measure: {}".format(metric))

    def _findThreshold(self, loudnesses, annotations, measure):
        """Finds the best loudness threshold for either harmonic or percussive
        classification.

        Args:
            loudnesses (list of float): The calculated loudnesses.
            annotations (list of bool): The annotated labels.
        
        Returns:
            float: The optimal threshold.
            float: The score achieved by the optimal threshold.

        """

        # Sort loudnesses and respective annotations
        sortedLoudnesses, sortedAnnotations = zip(
            *sorted(zip(loudnesses, annotations)))

        # Preparation
        scores = []
        loudnessesBelow = []
        loudnessesAbove = list(loudnesses)
        estimations = [True] * len(loudnesses)

        # Try out all reasonable thresholds
        for i in range(len(loudnesses) - 1):
            estimations[i] = False
            loudnessesBelow.append(loudnessesAbove.pop(0))
            scores.append(
                self._score(measure, estimations, sortedAnnotations,
                              loudnessesBelow, loudnessesAbove))

        # Find optimal threshold
        idx = np.argmax(scores)
        threshold = (sortedLoudnesses[idx] + sortedLoudnesses[idx + 1]) * 0.5
        bestScore = scores[idx]

        return threshold, bestScore

    @staticmethod
    def _flatten(listOfSignals):
        return [x for signal in listOfSignals for x in signal.values]

    def fit(self, X, y):
        # Store copy of parameters to restore state at the end
        parametersCopy = copy.deepcopy(self.parameters)

        # Flatten annotations
        annotationsHarmonic, annotationsPercussive = zip(*y)
        annotationsHarmonic = self._flatten(annotationsHarmonic)
        annotationsPercussive = self._flatten(annotationsPercussive)

        # Calculate all possible combinations of parameter values,
        # Example: val(a) in {1, 2} and val(b) in {10, 20}

        paramNames = self.parameters.keys()
        # ['a', 'b']

        possibleMappings = [
            param.fitEnumerator for param in self.parameters.values()
        ]
        # [ [1, 2], [10, 20] ]

        combinations = list(itertools.product(*possibleMappings))
        # [ (1, 10), (1, 20), (2, 10), (2, 20) ]

        # For each combination, find best possible threshold and score
        scores = []
        thresholds = []
        for combi in combinations:
            for i in range(len(paramNames)):
                self.setValue(paramNames[i], combi[i])

            loudnesses = []
            for inputs in X:
                loudnesses += self._calcLoudnesses(*inputs)
            loudnessesHarmonic, loudnessesPercussive = zip(*loudnesses)

            thresholdHarmonic, scoreHarmonic = self._findThreshold(
                loudnessesHarmonic, annotationsHarmonic,
                self.getValue("measureHarmonic"))
            thresholdPercussive, scorePercussive = self._findThreshold(
                loudnessesPercussive, annotationsPercussive,
                self.getValue("measurePercussive"))

            scores.append((scoreHarmonic, scorePercussive))
            thresholds.append((thresholdHarmonic, thresholdPercussive))

        # Gather results
        outputMatrix = np.column_stack((scores, thresholds, combinations))
        header = [
            "score harm.", "score perc.", "best threshold harm.",
            "best threshold perc."
        ] + paramNames

        # Restore former parameters
        self.parameters = parametersCopy

        return outputMatrix, header

    def evaluate(self, X, y):
        # Flatten input
        annotationsHarmonic, annotationsPercussive = zip(*y)
        annotationsHarmonic = self._flatten(annotationsHarmonic)
        annotationsPercussive = self._flatten(annotationsPercussive)

        # Calculate loudnesses
        loudnesses = []
        for inputs in X:
            loudnesses += self._calcLoudnesses(*inputs)

        # Obtain classifications and loudness groups
        classificationsHarmonic = []
        loudnessesAboveHarmonic = []
        loudnessesBelowHarmonic = []
        classificationsPercussive = []
        loudnessesAbovePercussive = []
        loudnessesBelowPercussive = []

        for loudnessHarmonic, loudnessPercussive in loudnesses:
            isHarmonic = loudnessHarmonic > self.getValue("thresholdHarmonic")

            classificationsHarmonic.append(isHarmonic)
            if isHarmonic:
                loudnessesAboveHarmonic.append(loudnessHarmonic)
            else:
                loudnessesBelowHarmonic.append(loudnessHarmonic)

            isPercussive = loudnessPercussive > self.getValue(
                "thresholdPercussive")

            classificationsPercussive.append(isPercussive)
            if isPercussive:
                loudnessesAbovePercussive.append(loudnessPercussive)
            else:
                loudnessesBelowPercussive.append(loudnessPercussive)

        # Calculate scores
        scoreHarmonic = self._score(
            self.getValue("measureHarmonic"),
            classificationsHarmonic,
            annotationsHarmonic,
            loudnessesBelowHarmonic,
            loudnessesAboveHarmonic
        )

        scorePercussive = self._score(
            self.getValue("measurePercussive"),
            classificationsPercussive,
            annotationsPercussive,
            loudnessesBelowPercussive,
            loudnessesAbovePercussive
        )

        return scoreHarmonic, scorePercussive
