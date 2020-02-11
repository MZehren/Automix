import numpy as np

from automix.rules.rule import Rule
from automix.featureExtraction.lowLevel import CoreFinder
from automix.model.classes.signal import Signal


class VeireTransitionRule(Rule):
    """
    The transition occurs on specific locations in the mix:
    double drop: on each track the transition is on a down to up

    A\B dd - du - ud - uu
    dd  
    du       x
    ud  x**        
    uu       x*

    * the track A needs a ud 16 bars after the uu
    ** the track B needs to start 16 bars before the cue
    """

    def run(self, mix, boundaries, switches):
        tracks = Rule.getTracks(mix, boundaries)
        # cf = CoreFinder(parameterIncludeBorders=False)
        # for A, B, switchPositions in [(tracks[i], tracks[i + 1], switches[i]) for i in range(len(switches))]:
        #     start, switch, stop = switchPositions

        #     # compute if it's a core in all the segments (start-switch and switch-stop) are full segments or not
        #     # TODO don't call the predict one, it's too slow
        #     aCore, bCore = [
        #         cf.predictOne(track.features["samples"],
        #                       Signal(1, times=[track.getTrackTime(start),
        #                                        track.getTrackTime(switch),
        #                                        track.getTrackTime(stop)]))[0] for track in [A, B]
        #     ]

        #     isDoubleDrop = not aCore[0] and aCore[1] and not bCore[0] and bCore[1]
        #     isRolling = aCore[0] and aCore[1] and not bCore[0] and bCore[1]  # TODO: implement the aCore[2] == False
        #     isRelaxed = aCore[0] and not aCore[1] and not bCore[0] and not aCore[1] #TODO: implement the aCore[0] == start of the track
        #     if isDoubleDrop or isRolling or isRelaxed:
        #         return 1
        # return 0
        if len(tracks) < 2:
            return 0
        scores = []
        for A, B, switchPositions in [(tracks[i], tracks[i + 1], switches[i]) for i in range(len(switches))]:
            start, switch, stop = switchPositions
            coreStartA, coreSwitchA, coreStopA = [
                A.features["core"].getValue(A.getTrackTime(time), toleranceWindow=0.1) for time in switchPositions
            ]
            coreStartB, coreSwitchB, coreStopB = [
                B.features["core"].getValue(B.getTrackTime(time), toleranceWindow=0.1) for time in switchPositions
            ]

            isDoubleDrop = not coreStartA and not coreStartB and coreSwitchA and coreSwitchB

            isRolling = coreStartA and coreSwitchA and not coreStopA and not coreStartB and coreSwitchB and coreStopB

            isRelaxed = coreStartA and not coreSwitchA and not coreStopA and not coreStartB and not coreSwitchB  #TODO and start of the song here
            if isDoubleDrop:
                self.description = "Double drop"
                scores.append(1)
            elif isRolling:
                self.description = "Rolling"
                scores.append(1)
            elif isRelaxed:
                self.description = "Relaxed"
                scores.append(1)
            else:
                scores.append(0)
        return np.mean(scores)

