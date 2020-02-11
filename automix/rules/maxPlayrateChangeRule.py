from rules.rule import Rule

class MaxPlayrateChangeRule(Rule):
    """
    if the playRate is affected by more than 20% it's going to sound bad
    returns 0 if one of the track's playrate is changed by 20% or more
    returns 1 if at most the tracks' playrate is changed by 5%
    """

    def __init__(self, maxPlayRateChange=0.2, minPlayRateChange=0.05):
        self.maxPlayRateChange = maxPlayRateChange
        self.minPlayRateChange = minPlayRateChange
        super(MaxPlayrateChangeRule, self).__init__()

    def run(self, mix, boundaries):
        tracks = Rule.getTracks(mix)
        change = max([abs(track.playRate - 1) for track in tracks])
        score = 1 if change < self.minPlayRateChange else 1 - \
            (change-self.minPlayRateChange)/(self.maxPlayRateChange-self.minPlayRateChange)
        return max(score, 0)

    def __str__(self):
        return "PlayRate"
