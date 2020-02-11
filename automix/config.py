"""
TODO: Move the functions to the correct location
"""
import logging as log
import os

DATASET_LOCATION = "/home/mickael/Documents/programming/dj-tracks-switch-points/"
CACHE_LOCATION = "../annotations/"
CACHE_LEVEL = 0
LOG_LEVEL = log.DEBUG

log.getLogger().setLevel(LOG_LEVEL)


def k_fold_split(X, Y, k=10, shuffleDataset=True):
    """
    Split both list X and Y into k folds
    random will shuffle the data before, so two calls would not return the same folds

    ex: print(k_fold_split(["A", "B", "C", "D", "E", "F", "G"], ["a", "b", "c", "d", "e", "f", "g"], k=3, shuffleDataset=0))
    [[('A', 'a'), ('B', 'b')], [('C', 'c'), ('D', 'd')], [('E', 'e'), ('F', 'f'), ('G', 'g')]]
    """
    from random import shuffle

    assert len(X) == len(Y) and k <= len(X)

    def chunkIt(seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    indexes = list(range(len(X)))
    if shuffleDataset:
        shuffle(indexes)

    foldsIndexes = chunkIt(indexes, k)
    folds = [[(X[i], Y[i]) for i in foldIndexes] for foldIndexes in foldsIndexes]
    return folds


def _getFilename(path):
    file, ext = os.path.splitext(os.path.basename(path))
    if ext != ".mp3" and ext != ".jams": # in case that we give a file without ext but still contain a "." in the name 
        return file + ext
    else:
        return file


def _getFileType(path):
    """
    return the extension of the file based on the path
    i.e.: 'MP3' or 'WAVE'
    """
    ext = path.split("/")[-1].split(".")[-1]
    if ext == "mp3":
        return 'MP3'
    if ext == "wav":
        return "WAVE"
    if ext == "jams":
        return "JAMS"
    else:
        return ext


def getFolderFiles(directory):
    """
    returns the paths located in this folder
    """
    paths = sorted(os.listdir(directory))
    knownTypes = ["MP3", "WAVE", "mp4", "m4a", "JAMS"]
    return [os.path.join(directory, path) for path in paths if _getFileType(path) in knownTypes]


def GET_PAOLO_FULL(checkCompletude=True, sets=["paolo1", "paolo2", "paolo3", "paolo4", "paolo5", "paolo6", "paolo7"]):
    """
    return the path of the audio files (.mp3) and the anotation files (.jams)
    if checkCompletude si True, erase the tracks without annotations and erase annotation without tracks
    """
    tracksPaths = []
    for set in sets:
        tracksPaths += getFolderFiles(DATASET_LOCATION + str(set) + "/audio/")

    gtTrackPaths = getFolderFiles(DATASET_LOCATION + "clean/annotations/")
    if checkCompletude:
        tracksPaths, gtTrackPaths = CHECK_COMPLETUDE(tracksPaths, gtTrackPaths)
    return tracksPaths, gtTrackPaths


def CHECK_COMPLETUDE(tracksPaths, gtTrackPaths):
    """
    Check if all the files are annotated and each annotation has a file
    """
    tracksPaths = sorted(tracksPaths, key=lambda x: _getFilename(x))
    gtTrackPaths = sorted(gtTrackPaths, key=lambda x: _getFilename(x))

    newTracksPaths = [track for track in tracksPaths if _getFilename(track) in [_getFilename(t) for t in gtTrackPaths]]
    newgtTrackPaths = [track for track in gtTrackPaths if _getFilename(track) in [_getFilename(t) for t in tracksPaths]]

    if len(newTracksPaths) != len(tracksPaths):
        log.info(("Becareful all the tracks are not annotated", len(newTracksPaths), len(tracksPaths)))
        log.debug("\n".join(
            [track for track in tracksPaths if _getFilename(track) not in [_getFilename(t) for t in gtTrackPaths]]))
        log.debug("\n".join(
            [track for track in gtTrackPaths if _getFilename(track) not in [_getFilename(t) for t in tracksPaths]]))

    return newTracksPaths, newgtTrackPaths
