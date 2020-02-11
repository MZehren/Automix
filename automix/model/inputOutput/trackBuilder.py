# """
# module to construct a track.
# it knows how to load a track from different serialization (json, jams) 
# or can load from a audiofile and call the feature extraction
# it also handles the cache system
# """
# import json
# import logging as log
# import os
# import sys
# import time
# from typing import List

# import numpy as np
# from pkg_resources import resource_filename

# from automix.featureExtraction.automaticDrumsTranscription import \
#     MadmomDrumsProxy
# from automix.featureExtraction.beats.madmomBeatDetection import \
#     MadmomBeatDetection
# from automix.featureExtraction.estimator import Estimator
# from automix.featureExtraction.harmonicPercussiveClassification import Hpss
# from automix.featureExtraction.lowLevel import (Cqt, Normalize, Pcp, PeakPicking, PeakSelection, Periodicity, Quantize, ReadFile,
#                                                 Windowing, ReplayGain, CoreFinder)
# from automix.featureExtraction.novelty import Checkerboard
# from automix.featureExtraction.vocalSeparation import VocalMelodyExtraction
# from automix.model.classes import Deck, Segment, Track
# from automix.model.inputOutput.serializer import myJSONEncoder
# from automix.model.inputOutput.serializer.reaperProxy import ReaperProxy


# class TrackBuilder(object):
#     """
#     Build a Track with all the estimators available in the feature extraction module
#     """

#     def __init__(self):
#         self.audioTypes = ["MP3", "WAVE", "mp4", "m4a"]
#         self.jamsTypes = ["JAMS"]
#         self.knownTypes = self.audioTypes + self.jamsTypes

#     def _getFilename(self, path):
#         """
#         return the name of the file from the path
#         """
#         # TODO create a proper database
#         return ".".join(path.split("/")[-1].split(".")[:-1])

#     def _getFileType(self, path):
#         """
#         return the extension of the file based on the path
#         i.e.: 'MP3' or 'WAVE'
#         """
#         ext = path.split("/")[-1].split(".")[-1]
#         if ext == "mp3":
#             return 'MP3'
#         if ext == "wav":
#             return "WAVE"
#         if ext == "jams":
#             return "JAMS"
#         else:
#             return ext

#     def getFolderPaths(self, directory):
#         """
#         returns the paths located in this folder
#         """
#         paths = sorted(os.listdir(directory))
#         return [directory + path for path in paths if self._getFileType(path) in self.knownTypes]

#     def getTracklistPaths(self, path):
#         """
#         returns the paths located in this json file
#         TODO: check if the file exist ?
#         """
#         with open(path, 'r') as tracklistFile:
#             paths = json.load(tracklistFile)
#         paths = [str("/".join(path.split("/")[:-1]) + "/" + track) for track in paths if track]
#         return paths

#     def getPeakWorkflow(self, feature, sparse=False, windowPanning=0, name=None, forceRefreshCache=False, addZeroStart=True):
#         """
#         Create the graph of nodes to retreive the peaks from any feature.
#         - You can specify if the feature is parse: the aggregation of the quantization will be based on the sum instead of the
#         RMSE
#         - You can specify the panning of the window of the quantization in percentage of a strongbeat
#         - You can specify a name. If not, the name will be the name of the feature
#         - You can forceRefreshCache to compute again cached features (to be used if any extractor has been updated)
#         """
#         if name is None:
#             name = feature
#         featureEstimators = [
#             Windowing(inputSamples=feature,
#                       inputGrid="strongBeats",
#                       output=name + "RMSE",
#                       parameterAggregation= "sum" if sparse else "rmse",
#                       parameterPanning=windowPanning,
#                       parameterSteps=1,
#                       forceRefreshCache=forceRefreshCache),
#             Normalize(inputSamples=name + "RMSE",
#                       outputNormalizedSamples=name + "RMSENormalized",
#                       forceRefreshCache=forceRefreshCache),
#             Checkerboard(
#                 inputSamples=name + "RMSENormalized",  # Checkerboard can be removed as it is replaced in getFeatureW
#                 outputNovelty=name + "Checkerboard",
#                 parameterAddZerosStart=addZeroStart,
#                 forceRefreshCache=forceRefreshCache,
#                 parameterWindowSize=16)  # parameterWindowSize=16*8, forceRefreshCache=False),
#         ]
#         return featureEstimators

#     def getFeatureW(self,
#                     features,
#                     topPeaks=[None],
#                     salienceThreshold=[0.4],
#                     relativeDistance=[1],
#                     inputSalience=["kickRMSENormalized", "harmonicRMSENormalized", "percussiveRMSENormalized"]):
#         """
#         Get estimators for the peak picking for the features and parameters in features 
#         """
#         w = []
#         # Add all the novelty curves and independent peak picking
#         for feature, parameters in features:
#             # c = Checkerboard(inputSamples=feature + "RMSENormalized", outputNovelty=feature + "Checkerboard")
#             # c.parameters["addZerosStart"].fitStep = parameters[0]  # [False, True]
#             # c.parameters["windowSize"].fitStep = parameters[1]  # [8,32]
#             pp = PeakPicking(inputSignal=feature + "Checkerboard", outputPeaks=feature + "CheckerboardPeaks")
#             pp.parameters["relativeThreshold"].fitStep = parameters[2]  # [0.1,0.3]
#             pp.parameters["minDistance"].fitStep = parameters[3]  # [8,32]
#             w += [pp]

#         # Compute the periodicity: 8 SB = 4 bars
#         p = Periodicity(inputFeatures=[feature + "Checkerboard" for feature, parameters in features])
#         p.parameters["period"].fitStep = [8]
#         p.parameters["featureAggregation"].fitStep = ["quantitative"]  # ["quantitative", "qualitative"]
#         p.parameters["distanceMetric"].fitStep = ["RMS"]  # ["RMS", "sum", "Veire"]
#         w.append(p)

#         # Quantize the beats to the periodicity
#         for feature, parameters in features:
#             q = Quantize(inputSignal=feature + "CheckerboardPeaks", outputSignal=feature + "Quantized")
#             q.parameters["maxThreshold"].fitStep = [0]
#             w += [q]

#         # Get the top peaks + minimum salience
#         ps = PeakSelection(inputPeaks=[feature + "Quantized" for feature, parameters in features],
#                            inputSalience=inputSalience,
#                            parameterMergeFunction=np.mean)
#         ps.parameters["absoluteTop"].fitStep = topPeaks
#         ps.parameters["salienceTreshold"].fitStep = salienceThreshold
#         ps.parameters["relativeDistance"].fitStep = relativeDistance
#         w.append(ps)
#         return w

#     def getAllEstimators(self, train=False, loadUncached=False):
#         """
#         train (False). If True return 2 list of estimators: 
#             - The basic estimators without parameters
#             - The trainable estimators with parameters to fit
        
#         loadUncached (False). If True, return in the list the feature extraction estimator which are not cached, 
#         but whose result is already used in more advances features

#         """
#         if loadUncached:
#             estimators = [
#                 ReadFile(),
#                 MadmomBeatDetection(),
#                 # VocalMelodyExtraction(),
#                 Hpss(),
#                 MadmomDrumsProxy(),
#                 Cqt(parameterBinNumber=84, parameterScale="Perceived dB", outputCqt="cqtPerceiveddB"),
#                 Pcp(parameterNieto=True),
#                 # Pcp(parameterNieto=False, outputPcp="chromagram")
#             ]
#         else:
#             estimators = [
#                 ReadFile(),
#                 MadmomBeatDetection(),
#                 # Pcp(parameterNieto=True),
#                 # ReplayGain(inputGrid=None),
#                 MadmomDrumsProxy()
#             ]

#         # estimators += self.getPeakWorkflow("samples")
#         estimators += self.getPeakWorkflow("pcp", addZeroStart=False)
#         # estimators += self.getPeakWorkflow("chromagram", addZeroStart=False)
#         estimators += self.getPeakWorkflow("cqtPerceiveddB", addZeroStart=False)
#         estimators += self.getPeakWorkflow("harmonic")
#         estimators += self.getPeakWorkflow("percussive")
#         estimators += self.getPeakWorkflow("kick", sparse=True, windowPanning=0.21)
#         estimators += self.getPeakWorkflow("snare", sparse=True, windowPanning=0.21)
#         estimators += self.getPeakWorkflow("hihat", sparse=True, windowPanning=0.21)

#         trainableEstimator = self.getFeatureW(
#             [
#                 # ("samples", [[False], [16], [0.3], [8]]),
#                 ("pcp", [[False], [16], [0.3], [8]]),
#                 # ("chromagram", [[False], [16], [0.3], [8]]),
#                 ("cqtPerceiveddB", [[False], [16], [0.3], [8]]),
#                 ("harmonic", [[True], [16], [0.3], [8]]),
#                 ("percussive", [[True], [16], [0.3], [8]]),
#                 ("kick", [[True], [16], [0.3], [8]]),
#                 ("snare", [[True], [16], [0.3], [8]]),
#                 ("hihat", [[True], [16], [0.3], [8]])
#             ],
#             inputSalience=["harmonicRMSENormalized"],
#             salienceThreshold=[.4],
#             topPeaks=[None],
#             relativeDistance=[1])

#         if train:
#             return estimators, trainableEstimator
#         else:
#             estimators += trainableEstimator
#             estimators += [CoreFinder()]
#             return estimators

#     def runEstimatorsOne(self,
#                          path: str,
#                          onlyCache: bool = False,
#                          estimators: List[Estimator] = None,
#                          cachingLevel: int = 0,
#                          cleanCache: bool = False) -> List[Track]:
#         """
#         run any list of estimators to extract features from an audio file.

#         PARAMETERS:
#             path (String or list<String>): path(s) to the audio file(s)
#             onlyCache (boolean): flag used to prevent new computation of features
#             estimators (list<Estimator>): Any sequence of estimators. Pay attention to the order of the estimators to make
#                 sure that an estimator expecting the output of another one appears later in the list
#             cachingLevel (int): Minimal cache level to serialize
#             cleanCache (bool): The cache is going to be used then flushed with the content of the current estimators ONLY
#         """
#         # get cache & build track
#         estimators = estimators if estimators is not None else self.getAllEstimators()
#         serializedFeatures = self.getCache(self._getFilename(path))
#         track = Track(name=self._getFilename(path), path=path, sourceType=self._getFileType(path))
#         track.features["path"] = path
#         serializedFeatures["path"] = {"value": path}

#         # Run all the estimator if not cached
#         for estimator in estimators:
#             timerStart = time.process_time()

#             outputNotCached = any([
#                 output not in serializedFeatures or serializedFeatures[output]["estimator"] != str(estimator)
#                 for output in estimator.outputs
#             ])
#             inputIsNew = any([
#                 input in track.features and input not in serializedFeatures for input in np.hstack(estimator.inputs)
#             ])  # the input is new if it was computed this round
#             if not onlyCache and (estimator.forceRefreshCache or outputNotCached
#                                   or inputIsNew):  # Run the estimator TODO: Use the Estimator.forwardPass ?
#                 log.debug("* " + str(estimator))
#                 outputs = estimator.predictOne(*[[track.features.get(f)
#                                                   for f in feature] if isinstance(feature, list) else track.features.get(feature)
#                                                  for feature in estimator.inputs])
#                 track.features.update({estimator.outputs[i]: outputs[i] for i in range(len(estimator.outputs))})

#             else:  # Retrieve from the cache
#                 log.debug("- " + str(estimator))
#                 track.features.update({
#                     feature: serializedFeatures[feature]["value"]
#                     for feature in estimator.outputs
#                     if feature in serializedFeatures and str(estimator) == serializedFeatures[feature]["estimator"]
#                 })

#             log.debug("  time :" + str(time.process_time() - timerStart))

#         # serialize
#         if not onlyCache:
#             toSerialize = {} if cleanCache else serializedFeatures
#             for estimator in [e for e in estimators if e.cachingLevel <= cachingLevel]:
#                 toSerialize.update(
#                     {output: {
#                         "estimator": str(estimator),
#                         "value": track.features[output]
#                     }
#                      for output in estimator.outputs})
#             self.setCache(self._getFilename(path), toSerialize)
#         return track

#     def runEstimators(self,
#                       path: str,
#                       onlyCache: bool = False,
#                       estimators: List[Estimator] = None,
#                       cachingLevel: int = 0,
#                       cleanCache: bool = False) -> List[Track]:
#         """
#         run any list of estimators to extract features from a list of files.
#         """
#         if isinstance(path, list):
#             results = []
#             for i, p in enumerate(path):
#                 log.info((i + 1, len(path), p))
#                 results.append(
#                     self.runEstimatorsOne(p,
#                                           onlyCache=onlyCache,
#                                           estimators=estimators,
#                                           cachingLevel=cachingLevel,
#                                           cleanCache=cleanCache))

#             return results

#     def getCache(self, filename):
#         """
#         Lookup the cached features of the files in the correct folder and deserialize the file
#         filename is the name of the file from self._getfilename
#         TODO: put the cached folder in the config
#         """
#         # cache lookup
#         cachePath = resource_filename(__name__, "../../../annotations/features/" + filename + ".json")
#         try:  # retrieve the cache
#             serializedFeatures = TrackBuilder.jsonDeserializeFeatures(cachePath)
#             return serializedFeatures
#         except Exception:
#             log.error(("jsonDeserializeFeatures error :", cachePath))

#     def setCache(self, filename, data):
#         """
#         Serialize the file in the correct folder depending on the cache level set
#         """
#         cachePath = resource_filename(__name__, "../../../annotations/features/" + filename + ".json")
#         TrackBuilder.jsonSerializeFeatures(cachePath, data)

#     # @staticmethod
#     # def jsonDeserializeFeatures(path):
#     #     """
#     #     instantiate a Track from the same encoding used by the toString method
#     #     """
#     #     serializedFeatures = {}
#     #     try:
#     #         with open(path) as file:
#     #             serializedFeatures = myJSONEncoder.decode(file.read())
#     #     except Exception:
#     #         pass

#     #     return serializedFeatures

#     # @staticmethod
#     # def jsonSerializeFeatures(path, features):
#     #     with open(path, 'w') as featuresFile:
#     #         featuresFile.write(json.dumps(features, cls=myJSONEncoder.MyJSONEncoder))

#     # @staticmethod
#     # def reaperSerialization(path: str, track: Track):
#     #     """
#     #     TODO: put that in reaperSerializer
#     #     write a reaper project to listen to the track
#     #     """
#     #     reaperProxy = ReaperProxy()
#     #     with open(path, 'w') as outfile:
#     #         outfile.write(
#     #             reaperProxy.getReaperProject(BPM=track.getTempo(),
#     #                                          markers=[
#     #                                              reaperProxy.getMarquer(time=track.features["boundaries"][i], label="")
#     #                                              for i in range(len(track.features["boundaries"]))
#     #                                          ],
#     #                                          decks=[Deck(tracks=[track])]))
