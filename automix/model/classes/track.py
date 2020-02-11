"""
Container for the class Track
"""
import collections
import copy
import logging as log
import os
import time
from typing import List

import librosa
import numpy as np
from pkg_resources import resource_filename

from automix import config
from automix.model.inputOutput.serializer.featureSerializer import \
    FeatureSerializer

# TODO:
# handle parameters training


class featuresGetter(dict):
    """
    Lazy loader of features
    Implements dict and call the estimator if the key is known but no value is cached.
    """

    def __init__(self, filename):
        """
        -filename: name/path of the track used to infer the path of the cache0
        """
        self._path = filename
        self["path"] = filename  # TODO This is not explicit and hard to understand
        self._estimators = self.getExecutionGraph()
        super().__init__()

    def getExecutionGraph(self):
        """
        Create the graph containing all the estimators usable and the link of dependency between them
        """
        # TODO: Do we use estimators instead of string for dependance ?
        # The dependance doesn't know which parameter was set for the input.
        from automix.featureExtraction.lowLevel.readFile import ReadFile, GetDuration
        from automix.featureExtraction.lowLevel.cqt import Cqt
        from automix.featureExtraction.lowLevel.pcp import Pcp
        from automix.featureExtraction.lowLevel.coreFinder import CoreFinder
        from automix.featureExtraction.lowLevel.replayGain import ReplayGain
        from automix.featureExtraction.lowLevel.onsetDetection import OnsetDetection
        from automix.featureExtraction.lowLevel.windowing import Windowing
        from automix.featureExtraction.structure.msafProxy import MsafProxy
        from automix.featureExtraction.beats.madmomBeatDetection import MadmomBeatDetection
        from automix.featureExtraction.harmonicPercussiveClassification.hpss import Hpss
        from automix.featureExtraction.vocalSeparation.vocalMelodyExtraction import VocalMelodyExtraction
        from automix.featureExtraction.automaticDrumsTranscription.madmomDrumsProxy import MadmomDrumsProxy
        import automix.featureExtraction.novelty as n
        import automix.featureExtraction.lowLevel as ll

        estimators: List[fe.Estimator] = [
            ReadFile(),
            GetDuration(),
            MadmomBeatDetection(parameterSnapDistance=0.05),
            VocalMelodyExtraction(),
            Hpss(),
            MadmomDrumsProxy(),
            Cqt(parameterBinNumber=84, parameterScale="Perceived dB", outputCqt="cqtPerceiveddB"),
            Pcp(parameterNieto=True),
            Pcp(parameterNieto=False, outputPcp="chromagram"),
            CoreFinder(),
            MsafProxy(algorithm="scluster", feature=None, outputSignal="msaf-scluster"),
            MsafProxy(algorithm="sf", feature="cqt", outputSignal="msaf-sf"),
            MsafProxy(algorithm="olda", feature=None, outputSignal="msaf-olda"),
            ReplayGain(inputGrid=None),
            OnsetDetection()
        ]

        def getPeakWorkflow(feature, sparse=False, windowPanning=0, forceRefreshCache=False, addZeroStart=True):
            """
            Create the graph of nodes to retreive the peaks from any feature.
            - You can specify if the feature is parse: the aggregation of the quantization will be based on the sum instead of the
            RMSE
            - You can specify the panning of the window of the quantization in percentage of a strongbeat
            - You can specify a name. If not, the name will be the name of the feature
            - You can forceRefreshCache to compute again cached features (to be used if any extractor has been updated)
            """
            name = feature
            featureEstimators = [
                Windowing(inputSamples=feature,
                          inputGrid="strongBeats",
                          output=name + "RMSE",
                          parameterAggregation="sum" if sparse else "rmse",
                          parameterPanning=windowPanning,
                          parameterSteps=1,
                          forceRefreshCache=forceRefreshCache),
                ll.Normalize(inputSamples=name + "RMSE",
                             outputNormalizedSamples=name + "RMSENormalized",
                             forceRefreshCache=forceRefreshCache),
                n.Checkerboard(
                    inputSamples=name + "RMSENormalized",  # Checkerboard can be removed as it is replaced in getFeatureW
                    outputNovelty=name + "Checkerboard",
                    parameterAddZerosStart=addZeroStart,
                    forceRefreshCache=forceRefreshCache,
                    parameterWindowSize=16)  # parameterWindowSize=16*8, forceRefreshCache=False),
            ]
            return featureEstimators

        estimators += getPeakWorkflow("samples")
        estimators += getPeakWorkflow("chromagram", addZeroStart=False)
        estimators += getPeakWorkflow("pcp", addZeroStart=False)
        estimators += getPeakWorkflow("cqtPerceiveddB", addZeroStart=False)
        estimators += getPeakWorkflow("harmonic")
        estimators += getPeakWorkflow("percussive")
        estimators += getPeakWorkflow("kick", sparse=True, windowPanning=0.21)
        estimators += getPeakWorkflow("snare", sparse=True, windowPanning=0.21)
        estimators += getPeakWorkflow("hihat", sparse=True, windowPanning=0.21)

        def getFeatureW(features,
                        topPeaks=[None],
                        salienceThreshold=[0.4],
                        relativeDistance=[1],
                        inputSalience=["kickRMSENormalized", "harmonicRMSENormalized", "percussiveRMSENormalized"]):
            """
            Get estimators for the peak picking for the features and parameters in features 
            """
            w = []
            # Add all the novelty curves and independent peak picking
            for feature in features:
                # c = Checkerboard(inputSamples=feature + "RMSENormalized", outputNovelty=feature + "Checkerboard")
                # c.parameters["addZerosStart"].fitStep = parameters[0]  # [False, True]
                # c.parameters["windowSize"].fitStep = parameters[1]  # [8,32]
                pp = ll.PeakPicking(inputSignal=feature + "Checkerboard", outputPeaks=feature + "CheckerboardPeaks")
                pp.parameters["relativeThreshold"].fitStep = [0.3]  # parameters[2]  # [0.1,0.3]
                pp.parameters["minDistance"].fitStep = [4]  # parameters[3]  # [8,32]
                w += [pp]

            # Compute the periodicity: 8 SB = 4 bars
            p = ll.Periodicity(inputFeatures=[feature + "Checkerboard" for feature in features])
            p.parameters["period"].fitStep = [8]
            p.parameters["featureAggregation"].fitStep = ["quantitative"]  # ["quantitative", "qualitative"]
            p.parameters["distanceMetric"].fitStep = ["RMS"]  # ["RMS", "sum", "Veire"]
            w.append(p)

            # Quantize the beats to the periodicity
            for feature in features:
                q = ll.Quantize(inputSignal=feature + "CheckerboardPeaks", outputSignal=feature + "Quantized")
                q.parameters["maxThreshold"].fitStep = [0]
                w += [q]

            # Get the top peaks + minimum salience
            ps = ll.PeakSelection(inputPeaks=[feature + "Quantized" for feature in features],
                                  inputSalience=inputSalience,
                                  parameterMergeFunction=np.mean)
            ps.parameters["absoluteTop"].fitStep = topPeaks
            ps.parameters["salienceTreshold"].fitStep = salienceThreshold
            ps.parameters["relativeDistance"].fitStep = relativeDistance
            w.append(ps)
            return w

        #"samples", "chromagram",
        estimators += getFeatureW(
            ["pcp", "cqtPerceiveddB", "harmonic", "percussive", "kick", "snare", "hihat"],
            inputSalience=["harmonicRMSENormalized"],
            salienceThreshold=[0.4],  #0.4 TODO: make that clear ?
            topPeaks=[None],
            relativeDistance=[1])

        # from automix.model.inputOutput.serializer import GraphvizSerializer
        # GraphvizSerializer().serializeEstimators(estimators)
        return estimators

    def __getitem__(self, key):
        return self.getItem(key)

    def getItem(self, key):
        """
        Returns the feature from RAM.
        If it's not here, it searchs it on disk and put it in RAM and returns it. 
        If it's not here, computes the feature, puts it on disk and RAM and returns it.

        The method is not pure, but it is so much easier like that
        """
        from automix import config

        # If an estimator has an input set to None
        if key is None:
            return

        # Search in RAM
        if self.__contains__(key) and super().__getitem__(key) is not None:  #TODO add estimator.forceRefreshCache == False?
            return super().__getitem__(key)

        # Search on disk
        estimator = self.getEstimator(key)  # Get the estimator known to compute this feature
        forceRefresh = estimator is not None and estimator.forceRefreshCache

        cache = self.getCache(key, estimator)
        if cache is not None and not forceRefresh:
            self[key] = cache
            # if config.CACHE_LEVEL >= estimator.cachingLevel:
            #     self.setCache(key, estimator, cache)
            return cache

        # Computes the feature
        computed = self.computeFeature(estimator)
        estimator.forceRefreshCache = False  #TODO this is to prevent multiple computation of a root estimator haveing multiple child
        for i, subKey in enumerate(estimator.outputs):  # an estimator has multiple outputs
            if config.CACHE_LEVEL >= estimator.cachingLevel:
                self.setCache(subKey, estimator, computed[i])
            self[subKey] = computed[i]
        return self[key]

    def computeFeature(self, estimator):
        """
        Retreive the feature by running the computation of the estimator
        The estimator executed might need more data which will run more estimators if needed
        """
        # Run the estimator by searching for the input
        # TODO this can create an infinite loop
        input = [[self.getItem(f) for f in feature] if isinstance(feature, list) else self.getItem(feature)
                 for feature in estimator.inputs]
        log.debug("computing " + str(estimator) + "->" + str(estimator.outputs) + " ...")
        timerStart = time.process_time()
        result = estimator.predictOne(*input)
        log.debug(str(np.round(time.process_time() - timerStart, decimals=2)) + "s")
        return result

    def getEstimator(self, key):
        """Get the estimator known to compute this feature"""
        estimator = [e for e in self._estimators if key in e.outputs]
        if len(estimator) > 1:
            raise Exception("graph of computation is not correct")
        elif len(estimator) == 0:
            return None
        return estimator[0]

    def propagateForceRefreshCache(self, estimator):
        """
        set the refresh to True for the estimator and all the following estimators later in the computational graph
        """
        estimator.forceRefreshCache = True
        for output in np.hstack(estimator.outputs):
            self.pop(output, None)
            for e in [e for e in self._estimators if output in np.hstack(e.inputs)]:
                self.propagateForceRefreshCache(e)  #TODO: Slow computation..

    def getCache(self, key, estimator):
        """
        retrieve the cached data of the feature for this track or return None
        """
        cachePath = self.getCachePath(key)
        cache = FeatureSerializer.deserialize(cachePath)
        if cache is not None and cache["estimator"] == str(estimator):
            return cache["value"]
        else:
            return None

    def setCache(self, key, estimator, value):
        """
        Set the data in cache. 
        Uses the feature's name and the estimator (with parameters) to discriminate the stored informaiton
        """
        cachePath = self.getCachePath(key)
        FeatureSerializer.serialize(cachePath, {"estimator": str(estimator), "value": value})

    def getCachePath(self, key):
        return resource_filename("automix", config.CACHE_LOCATION + config._getFilename(self._path) + "." + key + ".json")


class Track(object):
    '''
    Class used as a data structure associated with a track.
    We store all the metadata/features of the track here.
    Try to keep all the units in seconds or beats

    use the TrackBuilder to instantiate it
    '''

    def __init__(self, path="", name="", sourceType="", length=100, position=0, soffs=0, playRate=1, populateFeatures=True):
        # features extracted
        if populateFeatures:
            self.features = featuresGetter(path)  # features dictionnary from the MIR estimators

        # metadatas for the player
        self.path = path  # Path to the audio
        self.name = name if name != "" else config._getFilename(path)  # name of the music
        self.sourceType = sourceType if sourceType else self.getSourceType()  # sourceType for Reaper (see Track.getFileType)
        # length of the track actually played by the player (it can be less or more than the duration)
        self.length = length
        # location of the item in the deck (global timestamp)
        self.position = position
        self.soffs = soffs  # Start in source
        self.playRate = playRate  # playback rate in ratio. 0.5 is half the speed
        self.preservePitch = 0  # boolean saying if the pitch should be preserved when changing the playback rate
        # fade in and out of the track without using the deck effects NOT supported by everything.
        self.fadeIn = 0.01
        self.fadeOut = 0.01
        self.stretchMarkers = []  # list of tuples of the shape (targetTrackTime, originalTrackTime)
        self.stretchMarkersProgressive = False  # tells if the stretch

        # effects for the deck
        # The FX should contain Points such as:
        self.FX = {"gainPt": [], "volPt": [], "lowPt": [], "midPt": [], "highPt": [], "hpfPt": [], "lpfPt": []}

    def __repr__(self):
        return self.name

    @staticmethod
    def getFilename(path):
        """
        return the name of the file from the path
        """
        # TODO create a proper database
        raise DeprecationWarning()
        return os.path.splitext(os.path.basename(path))[0]

    def __deepcopy__(self, memo):
        newone = copy.copy(self)
        # Do not deep copy the features extracted. only the FX for this instance of the track
        newone.FX = copy.deepcopy(self.FX, memo)
        return newone

    def jsonEncode(self):
        """
        returns a json encodable object for the json module
        """
        jsonSerializableObject = dict(self.__dict__)
        del jsonSerializableObject["FX"]
        del jsonSerializableObject["position"]
        del jsonSerializableObject["length"]
        del jsonSerializableObject["soffs"]
        del jsonSerializableObject["playRate"]
        del jsonSerializableObject["fadeIn"]
        del jsonSerializableObject["fadeOut"]
        return jsonSerializableObject

    def synchronize(self, tempo):
        """
        Change the playrate to match the provided tempo or a multiple
        """
        octaveTempi = [tempo / 2, tempo / 1.5, tempo, tempo * 1.5, tempo * 2]
        closestTempo = octaveTempi[np.argmin([abs(targetTempo - self.getTempo()) for targetTempo in octaveTempi])]
        self.playRate = closestTempo / self.getTempo()

    def jumpStart(self, time):
        """
        Make the music start at a specific second
        """
        self.soffs = time
        self.length = self.getDuration() / self.playRate - time

    def getStartPosition(self):
        return self.position

    def getSourceType(self):
        if ".wav" in self.path:
            return "WAVE"
        elif ".mp3" in self.path:
            return "MP3"
        elif ".ogg" in self.path:
            return "VORBIS"
        elif ".midi" in self.path:
            return "MIDI"

        return ""

    def getEndPosition(self):
        """returns the global time where the track stops"""
        return self.getDeckTime(self.length)

    def getSubdivision(self, subdivision=8):
        """
        returns the subdivision of the beat. ie 8th notes
        """
        if subdivision > 4:
            stepsBetweenBeats = subdivision // 4
            beats = self.getBeats().times
            return [
                beats[i] + (float(step) * ((beats[i + 1] - beats[i]) / stepsBetweenBeats)) for i in range(len(beats) - 1)
                for step in range(stepsBetweenBeats)
            ]
        else:
            raise NotImplementedError()

    def positionToBeat(self, position, toleranceWindow=0):
        """
        returns the index of the closest beat to the position specified (in track domain, without any playrate change).
        if the position difference to the closest beat is more than the tolerance Window returns none.
        TODO: Could be implemented as a reverse lookup table if needded
        """
        beats = self.getBeats()
        closestBeat = np.argmin([abs(beat - position) for beat in beats])
        if abs(beats[closestBeat] - position) <= toleranceWindow:
            return closestBeat
        return None

    def getFeature(self, featureName):
        """
        Check if a feature is computed before returning it
        """
        return self.features[featureName]

    def getTempo(self):
        """
        returns the tempo of the track
        """
        return self.getFeature("tempo")

    def getReplayGain(self):
        """
        returns the replayGain of the track
        """
        return self.getFeature("replayGain")[0]

    def getDuration(self):
        """
        returns the duration of the track
        """
        return self.getFeature("duration")  #TODO: change that

    def getBeats(self):
        """
        returns an array of timestamps for all the beats
        """
        return self.getFeature("beats")

    def getDownbeats(self):
        """
        returns an array of timestamps for all the downbeats
        """
        return self.getFeature("downbeats")

    def getCueIns(self):
        """
        Returns a SparseSignal containing the list of positions available as cues
        """
        return self.getFeature("selectedPeaks")

    def getSampleRate(self):
        return self.getFeature("sampleRate")

    def getKey(self):
        return self.getFeature("key")

    def getHarmonic(self):
        return self.getFeature("harmonic")

    def getPercussive(self):
        return self.getFeature("percussive")

    def getBarSamples(self):
        return self.getFeature("barSamples")

    def getLoudness(self):
        return self.getFeature("loudness")

    def getDeckTimes(self, trackTimes, enableExceeding=True, exceedingTolerance=0):
        """
        call getDeckTime for all elements in trackTimes
        """
        assert isinstance(trackTimes, collections.Iterable)
        results = [
            self.getDeckTime(time, enableExceeding=enableExceeding, exceedingTolerance=exceedingTolerance) for time in trackTimes
        ]
        return [result for result in results if result != -1]

    def getDeckTime(self, trackTime, enableExceeding=True, exceedingTolerance=0):
        """
        Convert from a local time to a global time based on the location of the track in the deck and the playrate
        the local time should be a float or an array of floats in seconds.
        return -1 if enableExceeding is False and the trackTime is not within the actual played item in the client.
        if enableExceeding is False, the exceedingTolerance parameter (in seconds)
        is here to return values even if the tracktime is not in the played window of the track, as long as it's not exceeding the tolerance
        """
        if enableExceeding or (trackTime > self.soffs - exceedingTolerance and trackTime <
                               (self.length * self.playRate) + self.soffs + exceedingTolerance):
            return self.position + (float(trackTime) - self.soffs) / self.playRate
        return -1

    def getTrackTime(self, deckTime):
        """
        Convert from a global time to a local time based on the location of the track in the deck and the playrate
        decktime should be a float in seconds or an array of floats
        """
        if isinstance(deckTime, list):
            return [self.getTrackTime(time) for time in deckTime]
        return (float(deckTime) - self.position) * self.playRate + self.soffs

    def applyEffects(self, signal):
        """
        Apply the volume and gain effects to values
        """
        signal = copy.deepcopy(signal)
        values = signal.values
        locations = signal.getTimes()
        for i, location in enumerate(locations):
            for effect in [self.FX["volPt"], self.FX["gainPt"]]:
                if not effect:
                    continue
                boundarieA = np.argmin([np.abs(location - pt.position) for pt in effect])
                boundarieB = boundarieA - \
                    1 if location < effect[boundarieA].position else boundarieA + 1
                dbAmplitude = 0
                if boundarieB < 0 or boundarieB >= len(effect):
                    # TODO:we don't know what is the other boundarie, let's assume that there is None
                    dbAmplitude = effect[boundarieA].amplitude
                # elif effect[boundarieA].shape == 0: # other interpolation
                else:  # linear interpolation
                    weight = np.abs(location - effect[boundarieA].position) / np.abs(effect[boundarieA].position -
                                                                                     effect[boundarieB].position)
                    dbAmplitude = ((1 - weight) * effect[boundarieA].amplitude) + (weight * effect[boundarieB].amplitude)
                    continue

                values[i] *= librosa.core.db_to_amplitude(dbAmplitude)
        return signal
