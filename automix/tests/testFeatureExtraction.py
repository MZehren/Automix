import unittest

import numpy as np
from scipy.stats import hmean

from automix import config
from automix.featureExtraction.lowLevel import (CoreFinder, PeakPicking, PeakSelection, Windowing)
from automix.featureExtraction.structure.eval import evalCuesMutliple
from automix.model.classes import (DenseSignal, Signal, SparseSegmentSignal, SparseSignal, Track)
from automix.model.inputOutput.serializer import JamsSerializer


class TestFeatureExtraction(unittest.TestCase):
    def testPeakPicking(self):
        # Basic test
        myInput = Signal([1.5, 1, 0, 0, 0, 2, 0], sampleRate=1)
        pp = PeakPicking(parameterMinDistance=2, parameterRelativeThreshold=0.2)
        result = pp.predictOne(myInput)[0]
        self.assertEqual(list(result.values), [1.5, 2.])
        self.assertEqual(list(result.times), [0, 5])

    def testPeakSelection(self):
        # Basic test
        myInput = [Signal(1, times=[0, 5, 10]), Signal(0.5, times=[0, 10])]
        ps = PeakSelection()
        result = ps.predictOne(myInput, None, None)[0]
        self.assertEqual(list(result.values), [1.5, 1, 1.5])
        self.assertEqual(list(result.times), [0, 5, 10])

        # Test with mean
        ps = PeakSelection(parameterMergeFunction=np.mean)
        result = ps.predictOne(myInput, None, None)[0]
        self.assertEqual(list(result.values), [0.75, 1, 0.75])
        self.assertEqual(list(result.times), [0, 5, 10])

    def testStructureEval(self):
        # basic tests
        result = evalCuesMutliple([Signal(1, times=[1, 3, 2]), Signal(1, times=[1, 2, 3])],
                                  [Signal(1, times=[1, 2, 3]), Signal(1, times=[1, 2])])
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["precision"], 5 / 6)
        self.assertEqual(result["fMeasure"], hmean([1, 5 / 6]))

        result = evalCuesMutliple([Signal(1, times=[1, 2, 4, 5, 6])], [Signal(1, times=[1, 2, 3, 4])], limitSearchSpace=True)
        self.assertEqual(result["recall"], 3 / 4)
        self.assertEqual(result["precision"], 3 / 3)

        result = evalCuesMutliple([Signal(1, times=[1, 2, 5, 6]), Signal(1, times=[1, 2, 3, 4])],
                                  [Signal(1, times=[1, 2, 3, 4]), Signal(1, times=[1, 2, 5, 6])],
                                  averagePerDocument=True,
                                  returnDetails=True,
                                  limitSearchSpace=True)
        self.assertEqual(result["recall"], [0.5, 0.5])
        self.assertEqual(result["precision"], [1, 0.5])
        self.assertEqual(result["fMeasure"], list(hmean([[0.5, 0.5], [1, 0.5]], axis=0)))

        # test duration
        result = evalCuesMutliple([Signal(1, times=[1, 2, 3])], [SparseSegmentSignal(1, [[1, 2], [1.6, 2.2]])])
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["precision"], 2 / 3)

    def testSignal(self):
        # test sort of the values
        signal = Signal([5, 4, 3, 2, 1], times=[5, 4, 3, 2, 1])
        self.assertEqual(list(signal.values), [1, 2, 3, 4, 5])

        # test assertion of duplicate values
        # with self.assertRaises(AssertionError):
        signal = Signal([5, 4, 3, 2, 1], times=[5, 4, 3, 3, 1])

        # test Qantization: remove doubles
        signal = Signal(1, times=[0, 1.1, 2.1, 2.5, 2.6, 2.7, 3.1])
        grid = Signal(-1, times=list(range(5)))
        signal.quantizeTo(grid)
        # In case 1 value is exactly between 2 grid ticks, use the smallest one
        self.assertEqual(list(signal.values), [1, 1, 2, 3])

        # test Quantization: maxThreshold for out of bound
        signal = Signal(1, times=[0, 1.1, 2.0, 2.1, 2.6, 2.7, 3.1])
        signal.quantizeTo(grid, maxThreshold=0.2)
        self.assertEqual(list(signal.times), [0, 1, 2, 3])
        self.assertEqual(list(signal.values), [1, 1, 2, 1])

        # test quantization: don't remove duplicates
        signal = Signal(1, times=[0, 1.1, 2.0, 2.1, 2.6, 2.7, 3.1])
        signal.quantizeTo(grid, maxThreshold=0.2, removeDuplicatedValues=False)
        self.assertEqual(list(signal.times), [0, 1, 2, 2, 3])
        self.assertEqual(list(signal.values), [1, 1, 1, 1, 1])

        # test quantization: don't remove out of bounds
        signal = Signal(1, times=[0, 1.1, 2.0, 2.1, 2.6, 2.7, 3.1])
        signal.quantizeTo(grid, maxThreshold=0.2, removeDuplicatedValues=False, removeOutOfBound=False)
        self.assertEqual(list(signal.times), [0, 1, 2, 2, 2.6, 2.7, 3])
        self.assertEqual(list(signal.values), [1, 1, 1, 1, 1, 1, 1])

        # test getIndex
        signal = Signal(1, times=[0, 1.1, 2.0, 2.1, 2.6, 2.7, 3.1])
        idx = signal.getIndex(2.2, toleranceWindow=0.5)
        self.assertEqual(idx, 3)

        # test clusterSignals
        result = Signal.clusterSignals([SparseSignal(1, [0, 1, 2, 3]), SparseSignal(0, [1, 3, 3.1])], minDistance=0, mergeValue=np.mean)
        self.assertEqual(list(result.times), [0, 1, 2, 3, 3.1])
        self.assertEqual(list(result.values), [1, 0.5, 1, .5, 0])
        

    def testSparseSegmentSignal(self):
        signal = SparseSegmentSignal(1, [[0, 1], [1, 2]])
        idx = signal.getIndex(2.2, toleranceWindow=0.5)
        self.assertEqual(idx, 1)

        idx = signal.getIndex(1.2, toleranceWindow=0.5)
        self.assertEqual(idx, 1)

    def testWindow(self):
        # RMS value normalize by sample number
        w = Windowing()
        input = Signal(2, times=list(range(20)))
        result, = w.predictOne(input, Signal(0, times=[-1, 2.1, 3.1, 4.1, 20]))
        self.assertEqual(result[0], result[2])

    def testJamsSerialization(self):
        aggregatedSignal = JamsSerializer.aggregateAnnotations([[{
            "time": 0,
            "duration": 0,
            "confidence": 0
        }], [{
            "time": 0,
            "duration": 5,
            "confidence": 0
        }]])
        self.assertEqual(len(aggregatedSignal), 1)

    def testCoreFinder(self):
        cf = CoreFinder(parameterIncludeBorders=False)
        values = DenseSignal(range(10), 1)
        grid = SparseSignal(1, [0, 5, 10])
        result = cf.predictOne(values, grid)[0]
        self.assertEqual(result.values.tolist(), [False, True])
        self.assertEqual(result.times.tolist(), [[0, 5], [5, 10]])

        values = SparseSignal(range(10), list(range(10)))
        grid = SparseSignal(1, [0, 5, 10])
        result = cf.predictOne(values, grid)[0]
        self.assertEqual(result.values.tolist(), [False, True])
        self.assertEqual(result.times.tolist(), [[0, 5], [5, 10]])

    def testCheckerboard(self):
        """Check the addZerosStart which is not implemented yet. Needs to be updated."""
        from automix.featureExtraction.novelty import Checkerboard
        c = Checkerboard()

        values = DenseSignal(np.random.rand(100, 5), 1)
        novelty1 = Checkerboard(parameterAddZerosStart=True).predictOne(values)[0]
        novelty2 = Checkerboard(parameterAddZerosStart=0).predictOne(values)[0]
        novelty3 = Checkerboard(parameterAddZerosStart=None).predictOne(values)[0]
        self.assertGreater(novelty2[0], novelty1[0])
        self.assertEqual(novelty3[0], 0)

    def testResults(self):
        """Check that the precision is not going down with update of the code (takes time to execute)"""
        tp, gttp = config.GET_PAOLO_FULL(checkCompletude=True)
        tracks = [Track(path=path) for path in tp]
        gttracks = [JamsSerializer.deserializeTrack(track, agreement=0.5) for track in gttp]
        result = evalCuesMutliple([track.features["selectedPeaks"] for track in tracks],
                         [track.features["switchIn"] for track in gttracks],
                         limitSearchSpace=True)
        self.assertGreater(result["fMeasure"], 0.6538)


if __name__ == '__main__':
    unittest.main()
