import logging as log
import subprocess

import numpy as np
from pkg_resources import resource_filename

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class MadmomDrumsProxy(Estimator):
    """
    call madmom version http://ifs.tuwien.ac.at/~vogl/dafx2018/ from Richard Vogl
    returns the time stamp of drum events.
    the drums id are :
    0: kick
    1: snare ?
    2: hi hat ?
    """

    def __init__(self,
                 parameterModel="CRNN_3",
                 inputPath="path",
                 outputKick="kick",
                 outputSnare="snare",
                 outputHihat="hihat",
                 cachingLevel=0,
                 forceRefreshCache=False):
        super().__init__()
        self.parameters = {"model": Parameter(parameterModel)}
        self.inputs = [inputPath]
        self.outputs = [outputKick, outputSnare, outputHihat]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, path: str):
        # TODO: Is it possible to install both version of madmom ?
        # args = ["ls", "-l"]
        args = [
            resource_filename(__name__, "../../../vendors/madmomDrumsEnv/bin/python"),
            resource_filename(__name__, "../../../vendors/madmom-0.16.dev0/bin/DrumTranscriptor"), "-m",
            self.parameters["model"].value, "single", path
        ]  # Calling python from python, Yay...
        process = subprocess.Popen(args, stdout=subprocess.PIPE)
        output = process.stdout.read().decode()

        # TODO read  stderr=subprocess.STDOUT
        # err = process.stderr.read().decode()
        # if err:
        #     log.error(err)

        result = [event.split("\t") for event in output.split("\n") if event]
        result = [row for row in result if len(row) == 2 and self.is_number(row[0]) and self.is_number(row[1])]
        kicks = [float(row[0]) for row in result if row[1] == "35" or row[1] == "0"]
        snares = [float(row[0]) for row in result if row[1] == "38" or row[1] == "1"]
        hihats = [float(row[0]) for row in result if row[1] == "42" or row[1] == "2"]

        return (Signal(np.ones(len(kicks)), times=kicks,
                       sparse=True), Signal(np.ones(len(snares)), times=snares,
                                            sparse=True), Signal(np.ones(len(hihats)), times=hihats, sparse=True))

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False
