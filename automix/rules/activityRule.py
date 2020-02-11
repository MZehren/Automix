from automix.rules.rule import Rule
from automix.utils.quantization import separateInEvenGrids
from automix.model.classes.signal import Signal


class ActivityRule(Rule):
    """
    some tracks contain empty segments (such as only vocal tracks). Make sure that those segments are actually overlaped with sound
    """

    # def runOne(self, trackA, trackB, boundaries):
    #     for track in [trackA, trackB]:
    #         volume = track.features["barMSE"].getValues(*boundaries)

    def run(self, mix, boundaries):
        tracks = Rule.getTracks(mix, boundaries)
        noiseThreshold = 0.1
        silenceRatio = 0.1

        masterSignal = Signal([], times=[])
        for track in tracks:
            postFXSignal = track.applyEffects(track.getFeature("barMSE"))
            postFXSignal.times = track.getDeckTime(postFXSignal.times)
            masterSignal.addSignal(postFXSignal)

        values = masterSignal.getValues(*boundaries)
        proportion = float(len([value for value in values if value < noiseThreshold])) / len(values)
        if proportion > silenceRatio:
            return 1-proportion
        else:
            return 1

    def __str__(self):
        return "Minimum Activity"
