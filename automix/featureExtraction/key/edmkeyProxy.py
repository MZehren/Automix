"""
This module is an adaptor for edmkey (https://github.com/angelfaraldo/edmkey).
"""

import os

# edmkey is not a package. adjusting PYTHONPATH is necessary

try:
    from vendors.edmkey import edmkey, evaluation
except ImportError:
    raise ImportError(
        "Couldn't import edmkey. The edmkey repository might be missing. Use\n"
        "git submodule init\n"
        "git submodule update.")


def setParameters(parameters):
    """Sets the constants in edmkey.py according to the provided parameter
    dictionary.

    Args:
        parameters: The dictionary containint the parameters.

    Sets:
        The constants in edmkey.py (too many to list explicitly).

    """

    def getValue(key):
        return parameters[key].value

    # file settings
    edmkey.SAMPLE_RATE = getValue("sampleRate")
    edmkey.VALID_FILE_TYPES = set(getValue("validFileTypes"))
    # analysis parameters
    edmkey.HIGHPASS_CUTOFF = getValue("highpassCutoff")
    edmkey.SPECTRAL_WHITENING = getValue("spectralWhitening")
    edmkey.DETUNING_CORRECTION = getValue("detuningCorrection")
    edmkey.DETUNING_CORRECTION_SCOPE = getValue("detuningCorrectionScope")
    edmkey.PCP_THRESHOLD = getValue("PCPThreshold")
    edmkey.WINDOW_SIZE = getValue("windowSize")
    edmkey.HOP_SIZE = getValue("hopSize")
    edmkey.WINDOW_SHAPE = getValue("windowShape")
    edmkey.MIN_HZ = getValue("minHz")
    edmkey.MAX_HZ = getValue("maxHz")
    edmkey.SPECTRAL_PEAKS_THRESHOLD = getValue("spectralPeaksThreshold")
    edmkey.SPECTRAL_PEAKS_MAX = getValue("spectralPeaksMax")
    edmkey.HPCP_BAND_PRESET = getValue("HPCPBandPreset")
    edmkey.HPCP_SPLIT_HZ = getValue("HPCPSplitHz")
    edmkey.HPCP_HARMONICS = getValue("HPCPHarmonics")
    edmkey.HPCP_NON_LINEAR = getValue("HPCPNonLinear")
    edmkey.HPCP_NORMALIZE = getValue("HPCPNormalize")
    edmkey.HPCP_SHIFT = getValue("HPCPShift")
    edmkey.HPCP_REFERENCE_HZ = getValue("HPCPReferenceHz")
    edmkey.HPCP_SIZE = getValue("HPCPSize")
    edmkey.HPCP_WEIGHT_WINDOW_SEMITONES = getValue("HPCPWeightWindowSemitones")
    edmkey.HPCP_WEIGHT_TYPE = getValue("HPCPWeightType")
    # key detector method
    edmkey.KEY_PROFILE = getValue("keyProfile")
    edmkey.USE_THREE_PROFILES = getValue("useThreeProfiles")
    edmkey.WITH_MODAL_DETAILS = getValue("withModalDetails")


def estimateKey(path, separator=" ", tempOutputFile="edmkey.temp"):
    """Estimates the key of a given audio file.

    Args:
        path (str): The absolute path to the audio file.
        separator (str): The separating char between key name and mode.
        tempOutputFile (str): The relative or absolute path to a temporary file,
            that will be created and directly deleted afterwards.

    Returns:
        str: The key of the audio file.

    """

    # estimate key. create a file with the key as side effect
    key = edmkey.estimate_key(path, tempOutputFile)
    # delete this file
    os.remove(tempOutputFile)
    # replace the default separator and return key
    return key.replace("\t", separator)


def convertKey(keyString):
    """Converts a key given as string to a formal representation as pair of
    integers (for name and mode of the key).

    Args:
        keyString (str): The key as string.

    Returns:
        list of int: The formal representation of the key.

    """

    return evaluation.key_to_list(keyString)


def mirexScore(estimation, groundtruth):
    """Calculates the mirex score of an estimated key given the correct key.

    Args:
        estimation (list of int): The estimated key in formal representation.
        groundtruth (list of int): The correct key in formal representation.

    Returns:
        float: The mirex score.

    """

    return evaluation.mirex_score(estimation, groundtruth)