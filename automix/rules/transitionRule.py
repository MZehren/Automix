import numpy as np

from automix.rules.rule import Rule


class TransitionRule(Rule):
    """
    Abstract superclass for all rules regarding the pair-wise transition
    between songs of a mix.
    """

    @staticmethod
    def score(trackA, trackB):
        """Calculates the score of two transitioning tracks. To be implemented
        by any subclass.

        Args:
            trackA (Track): The track that transitions into the second track.
            trackB (Track): The track into that the first track transitions.

        Returns:
            float: A score in [0, 1] grading the transition.

        """

        raise NotImplementedError("To be implemented")

    def run(self, mix, boundaries):
        """Calculates the score of the mix as the mean of the scores for all
        transitions between tracks. Overrides the method of the superclass.
        """

        # TODO update description?

        # sort out, which tracks of the mix transition into which tracks
        tracks = self.getTracks(mix, boundaries)
        tracks.sort(key=lambda x: x.getStartPosition())
        scores = []
        for i in range(len(tracks) - 1):
            trackA = tracks[i]
            for j in range(i + 1, len(tracks)):
                trackB = tracks[j]
                if trackA.getEndPosition() <= trackB.getStartPosition():
                    break
                # trackA transitions into trackB
                # compute score
                scores.append(self.score(trackA, trackB))

        if not scores:
            # no transitions detected. return best score
            return 1.0

        return np.mean(scores)