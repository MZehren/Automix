import mir_eval
import numpy as np


def evalTempoMIR(trackEvaluated, trackGT):
    return mir_eval.tempo.detection(np.array([trackGT.tempo, trackGT.tempo]), 0, np.array([trackEvaluated.tempo, trackEvaluated.tempo]), tol=0.1)


def evalTempoMadmom(trackEvaluated, trackGT, window=0.04):
    F1 = trackEvaluated.tempo > trackGT.tempo * \
        (1-window) and trackEvaluated.tempo < trackGT.tempo * (1+window)

    F2 = (trackEvaluated.tempo > trackGT.tempo * (1-window) and trackEvaluated.tempo < trackGT.tempo * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo/3 * (1-window) and trackEvaluated.tempo < trackGT.tempo/3 * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo/2 * (1-window) and trackEvaluated.tempo < trackGT.tempo/2 * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo*2 * (1-window) and trackEvaluated.tempo < trackGT.tempo*2 * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo*3 * (1-window) and trackEvaluated.tempo < trackGT.tempo*3 * (1+window)) 

    F3 = (trackEvaluated.tempo > trackGT.tempo * (1-window) and trackEvaluated.tempo < trackGT.tempo * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo/3 * (1-window) and trackEvaluated.tempo < trackGT.tempo/3 * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo/2 * (1-window) and trackEvaluated.tempo < trackGT.tempo/2 * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo*2 * (1-window) and trackEvaluated.tempo < trackGT.tempo*2 * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo*3 * (1-window) and trackEvaluated.tempo < trackGT.tempo*3 * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo*2/3 * (1-window) and trackEvaluated.tempo < trackGT.tempo*2/3 * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo*3/4 * (1-window) and trackEvaluated.tempo < trackGT.tempo*3/4 * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo*4/3 * (1-window) and trackEvaluated.tempo < trackGT.tempo*4/3 * (1+window)) \
        or (trackEvaluated.tempo > trackGT.tempo*5/4 * (1-window) and trackEvaluated.tempo < trackGT.tempo*5/4 * (1+window)) 
        
    return F1, F2, F3
