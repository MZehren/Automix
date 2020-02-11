import json

import jams
import numpy as np

from automix.model.classes.signal import Signal, SparseSegmentSignal, SparseSignal
from automix.model.classes.track import Track
from automix.model.inputOutput.serializer.serializer import Serializer


class JamsSerializer(Serializer):
    """
    class handling jams serialization of track or mixes
    """

    @staticmethod
    def aggregateAnnotations(annotations,
                             agreementThreshold=0,
                             agreementThresholdLastPoint=0.5,
                             distanceAgreement=0.5,
                             minimalAnnotator=0):
        """        
        Aggregate the points of the same track from different annotators.
        - Return only the points having a ratio of annotators annotating it above the threshold 
        - The points aggregated are clustered based on the distanceAgrement
        - The tracks without the threshold number of annotator are completely discarded (return None)
        - Only the points with a mean confidence from all the annotation above a threshold are kept
        
        Args:
        ----
            annotations ([type]): [description]
            agreementThreshold (int, optional): [description]. Defaults to 0.
            confidenceThreshold (int, optional): [description]. Defaults to 0.
            distanceAgreement (float, optional): [description]. Defaults to 0.5.
            minimalAnnotator (int, optional): [description]. Defaults to 0.
        
        Returns:
        -------
            [type]: [description]
        """
        allAnnotations = [annotation for annotator in annotations for annotation in annotator]

        clusteredPos = []  #each cluster of annotations
        clusteredIndex = set([])  #keep all the annotations already clustered
        for i, annotation in enumerate(allAnnotations):
            # Check if the position is already inserted, then don't add it twice:
            if i in clusteredIndex:
                continue

            # Cluster all values
            # TODO: this only cluster the segments based on their start. It might be interesting to merge the segments not starting at all locations
            clusterIdx = [
                j for j, value in enumerate(allAnnotations)
                if j not in clusteredIndex and abs(annotation["time"] - value["time"]) < distanceAgreement
            ]
            clusteredIndex.update(clusterIdx)

            # Compute the metrics for this annotations (the agreement, the mean position and the meanConfidence)
            meanTime = np.mean([allAnnotations[i]["time"] for i in clusterIdx])
            agreement = len(clusterIdx) / len(annotations)
            # confidence = np.mean([allAnnotations[i]["confidence"] for i in clusterIdx])
            duration = np.max([allAnnotations[i]["duration"] for i in clusterIdx])

            if agreement >= agreementThreshold:
                clusteredPos.append({
                    "start": meanTime,
                    "stop": meanTime + duration,
                    "agreement": agreement,
                    # "confidence": confidence
                })

        if agreementThresholdLastPoint:
            clusteredPos.sort(key=lambda x: x["start"])
            # Remove all the annotation below the agreementThresholdLastPoint after the last position above it
            lastPos = [i for i, pos in enumerate(clusteredPos) if pos["agreement"] >= agreementThresholdLastPoint]
            if len(lastPos) and clusteredPos[lastPos[-1]]["agreement"] >= agreementThresholdLastPoint:
                clusteredPos = clusteredPos[: lastPos[-1] + 1]
            else:
                print("ALERT:", clusteredPos)

        if len(annotations) >= minimalAnnotator:
            # check if duration is not 0. (take float imprecision into account)
            # if any([abs(c["start"] - c["stop"]) > 0.01 for c in clusteredPos]):
            #     return SparseSegmentSignal([a["agreement"] for a in clusteredPos],
            #                                [(a["start"], a["stop"]) for a in clusteredPos])
            # else:
            return SparseSignal([a["agreement"] for a in clusteredPos], [a["start"] for a in clusteredPos])

    @staticmethod
    def deserializeTrack(path, agreement=0.51, distanceAgreement=0.5, minimalAnnotator=0, minimalConfidence=0):
        """instantiate a Track from the jams encoding. https://github.com/marl/jams/
        
        Args:
        ----
            path (list[str]): path to the .JAMS file
            agreement (float, optional): minimal ratio of annotators agreeing to keep the point. Defaults to 0.51.
            distanceAgreement (float, optional): distance between annotations to cluster them to the same point. Defaults to 0.5.
            minimalAnnotator (int, optional): minimal number of annotators to keep the annotation. Defaults to 0.
            minimalConfidence (int, optional): minimal confidence to keep the annotation. Defaults to 0.
        
        Returns:
        -------
            Track: a track with annotations in it's features
        """
        reference = None
        track = Track()
        with open(path) as file:
            reference = json.load(file)

        # meta
        track.path = path
        track.features["duration"] = reference["file_metadata"]["duration"]
        track.name = reference["file_metadata"]["title"]

        switchsIn = []
        switchsOut = []
        for annotation in reference["annotations"]:
            # meta
            annotator = annotation["annotation_metadata"]["annotator"]["name"]
            # if annotator == "Marco":
            #     continue
            # old format segment_open
            if annotation["namespace"] == "segment_open":
                segments = annotation["data"]
                track.features["boundaries"] = Signal(1, times=[segment["time"] for segment in segments], sparse=True)
                track.features["labels"] = [segment["value"] for segment in segments]
            # tempo
            elif annotation["namespace"] == "tempo":
                track.features["tempo"] = annotation["data"][0]["value"]
            # Current format with confidence, segment, and multiple annotators
            elif annotation["namespace"] == "cue_point":
                segments = annotation["data"]
                switchsIn.append([segment for segment in segments if segment["value"]["label"] == "IN"])
                switchsOut.append([segment for segment in segments if segment["value"]["label"] == "OUT"])
                track.features["switchIn-" + annotator] = Signal(
                    1, times=[segment["time"] for segment in segments if segment["value"]["label"] == "IN"], sparse=True)

        track.features["switchIn"] = JamsSerializer.aggregateAnnotations(switchsIn,
                                                                         agreementThreshold=agreement,
                                                                         distanceAgreement=distanceAgreement,
                                                                         minimalAnnotator=minimalAnnotator)
        # track.features["switchOut"] = JamsSerializer.aggregateAnnotations(switchsOut,
        #                                                                   agreementThreshold=agreement,
        #                                                                   distanceAgreement=distanceAgreement,
        #                                                                   minimalAnnotator=minimalAnnotator)
        return track

    @staticmethod
    def serializeTrack(path, track: Track, features=[{"namespace": "beat", "data_source": "Madmom", 'feature': "beats"}]):
        """
        Serialize a track in jams format
        """
        jam = jams.JAMS()
        jam.file_metadata.duration = track.getDuration()
        for feature in features:
            annotation = jams.Annotation(namespace=feature["namespace"])
            annotation.annotation_metadata = jams.AnnotationMetadata(data_source=feature["data_source"])

            for t in track.getFeature(feature["feature"]):
                annotation.append(time=t, duration=0.0)

            jam.annotations.append(annotation)

        jam.save(path)
