"""
contains wave export with a possible click to sonify events in the tracks such as beats
"""
import numpy as np
from scipy.io import wavfile
import mir_eval

from automix.model.classes.track import Track

def sonifyClicks(ticks, audioHQ, sr, outputPath=None):
    """
     Put a click at each estimated beat in beats array
     todo: look at mir_eval which is used by msaf

     ticks can either be [time] or [[time,barPosition]]
    """
    # audioHQ, sr = Track.readFile(inputPath)
    # msaf.utils.sonify_clicks(audio_hq, np.array(tick), outputPath, sr)

    # Create array to store the audio plus the clicks
    # outAudio = np.zeros(len(audioHQ) + 100)

    # Assign the audio and the clicks
    outAudio = audioHQ
    if isinstance(ticks[0], list):
        audioClicks = getClick(
            [tick[0] for tick in ticks if tick[1] != 1],
            sr,
            frequency=1500,
            volume=0.8,
            length=len(outAudio))
        outAudio[:len(audioClicks)] += audioClicks

        audioClicks2 = getClick(
            [tick[0] for tick in ticks if tick[1] == 1],
            sr,
            frequency=1000,
            volume=1,
            length=len(outAudio))
        outAudio[:len(audioClicks2)] += audioClicks2
    else:
        audioClicks = mir_eval.sonify.clicks(ticks, sr) #getClick(ticks, sr, frequency=1500, length=len(outAudio))
        outAudio[:len(audioClicks)] += audioClicks

    # Write to file
    if outputPath:
        wavfile.write(outputPath, sr, outAudio)
    return outAudio


def getClick(clicks, fs, frequency=1000, offset=0, volume=1, length=0):
    """
    Generate clicks (this should be done by mir_eval, but its
    latest release is not compatible with latest numpy)
    """
    times = np.array(clicks) + offset

    # 1 kHz tone, 100ms with  Exponential decay
    click = np.sin(2 * np.pi * np.arange(fs * .1) * frequency /
                   (1. * fs)) * volume
    click *= np.exp(-np.arange(fs * .1) / (fs * .01))

    if not length:
        length = int(times.max() * fs + click.shape[0] + 1)

    return mir_eval.sonify.clicks(times, fs, click=click, length=length)