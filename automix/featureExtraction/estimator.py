"""
Container for the estimator class
"""
import copy
import inspect
import json
import logging as log
import os
from itertools import chain, combinations
from typing import List

import numpy as np


class Parameter(object):
    """
    Class encapsulating an estimator parameter
    It's usefull to normalize the fitting step

    Attributes:
        value (any): the value of this parameter
        fitEnumerator (list<any>): a list of values to try when fitting this parameter
    """

    def __init__(self, value, fitStep=None, minimum=None, maximum=None):
        self.minimum = minimum
        self.maximum = maximum
        self.value = value
        self.fitStep = fitStep

    @property
    def fitStep(self):
        return self._fitStep

    @fitStep.setter
    def fitStep(self, value):
        self._fitStep = value
        if value is not None:
            self._stepfitted = len(value) // 2
            self.value = self._fitStep[self._stepfitted]

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        """
        Force the values to be a numpy array
        """
        if self.maximum is not None and value > self.maximum:
            value = self.maximum
        if self.minimum is not None and value < self.minimum:
            value = self.minimum

        self._value = value

    def setFitValue(self, increment=1):
        """
        blabla
        """
        if self._stepfitted + increment < len(self.fitStep) and self._stepfitted + increment > 0:
            return False

        self._stepfitted += increment
        self.value = self.fitStep[self._stepfitted]
        return True

    def __str__(self):
        if callable(self.value
                    ):  #If the parameter's avlue is a function, return it's name instead of the default "<function hexValue>"
            return self.value.__name__
        else:
            return str(self.value)


class Estimator(object):
    """
    Base class for all estimators

    Attributes:
        parameters (Dict<Parameter>): The parameters of our estimator. Set at the instantiation.
        inputs (list<String>): List of the inputs of the predictOne method
        outputs (list<String>): List of the outputs of the predictOne method
        cachingLevel (int): Value specifying if the estimator benefits from caching
            0: cache
            1: cache only if necessary
            2: don't cache
        forceRefreshCache (bool): Make this node compute the data even if cache is available
        (in case the implementation is updated)
    """

    def __init__(self, parameters={}, inputs=[], outputs=[], cachingLevel=0, forceRefreshCache=False):
        """
        parameters: control how the prediction is going to be done
        inputs: Names/id of the features to use as the input of the prediction
            inputs can be a nested array if an input field has to be a list of values
        outputs: Names/id of the features to store the output.
        """
        self.parameters = {k: Parameter(v) for k, v in parameters.items()}
        self.inputs = inputs
        self.outputs = outputs
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def getInputsFlat(self):
        """
        Return the inputs after a horizontal stack applied to them
        [A, [B, C]] -> [A, B, C]
        """
        return np.hstack(self.inputs)

    def __str__(self):
        return type(self).__name__ + "{" + ", ".join(
            ["'" + key + "': '" + str(self.parameters[key]) + "'" for key in sorted(self.parameters.keys())]) + "}"

    def predict(self, T):
        """
        predict the result for the samples T

        Parameters
        ----------
            T (list<list<any>>): array-like, shape = (n_samples, n_features)
                Samples to classify.
        """
        return [self.predictOne(*t) for t in T]

    def predictOne(self, *features):
        """
        predict the result for one sample

        Parameters
        ----------
            features (any): arbitrary number of parameters needed by the estimator.
        """
        raise NotImplementedError()

    @staticmethod
    def powerSet(iterable: List, removeEmptySet=True):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        result = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
        if removeEmptySet:
            return list(result)[1:]
        else:
            return result

    @staticmethod
    def getGridSearchW(estimators: List["Estimator"]):
        """
        return all the possible combination of estimator parameters
        """
        # Get all the parameters
        params = [
            param for estimator in estimators for name, param in estimator.parameters.items()
            if param.fitStep is not None and len(param.fitStep)
        ]

        complexity = np.prod([len(p.fitStep) for p in params])
        if complexity > 1000:
            print(("complexity is too big", complexity))
            return []
        # Get all the values combinations (power set)
        # Works by building the list of tuples column wise.
        # for each known tupleValue x (starting at []), add y new entries for each new parameter
        #   if tupleValues = [[x1],[x2]] and the parameter = [y1,y2]
        #   newTuple = [[x1,y1], [x2,y1], [x1,y2], [x2,y2]]
        tupleValues = [[]]
        for parameter in params:
            newTuple = []
            for value in parameter.fitStep:
                for tuple in tupleValues:
                    newTuple.append(tuple + [(parameter, value)])
            tupleValues = newTuple

        return tupleValues

    @staticmethod
    def forwardPass(X: List["dict"], estimators: List["Estimator"]):
        """
        run all the estimators on the list of tracks
        """
        for x in X:
            for estimator in estimators:
                outputs = estimator.predictOne(*[[x[f] for f in feature] if isinstance(feature, list) else x[feature]
                                                 for feature in estimator.inputs])

                x.update({estimator.outputs[i]: outputs[i] for i in range(len(estimator.outputs))})

        return X

    @staticmethod
    def semiGridSearch(X, Y, Lw):
        """
        Do a grid search for each list of estimators in the list Lw sequentially
        it's a row search?
        """
        for w in Lw:  # set each group of parameters
            tuples = Estimator.getGridSearchW(w)
            scores = []
            for i, tuple in enumerate(tuples):
                for param, value in tuple:
                    param.value = value

                score = eval(X, Y, Lw)
                scores.append([score, tuple])
                print(i, end=' ')

            scores = sorted(scores, key=lambda x: x[0]["fMeasure"], reverse=True)
            setTuple(scores[0][1])
            print(scores[0][0])
        return scores[0]

    @staticmethod
    def gridSearch(X, Y, W, evalFunction, targetFunction="fMeasure"):
        """
        Compute the score of all the possible combination of parameter's value
        """
        scores = []
        tuples = Estimator.getGridSearchW(W)

        for i, tuple in enumerate(tuples):
            for param, value in tuple:
                param.value = value
            Y_ = Estimator.forwardPass(X, W)
            score = evalFunction(Y_, Y)
            log.info("grid search " + str(i) + "/" + str(len(tuples)))
            log.debug([score, tuple])
            scores.append([score, tuple])

        scores = sorted(scores, key=lambda x: x[0][targetFunction], reverse=True)
        # Update the parameters
        for param, value in scores[0][1]:
            param.value = value
        return scores

    def rowSearch(X, Y, W, evalFunction, steps=3, targetFunction="fMeasure"):
        oldScore = 0
        for i in range(steps):
            Y_ = Estimator.forwardPass(X, W)
            baseScore = evalFunction(Y_, Y)
            if oldScore == baseScore:
                break
            oldScore = baseScore
            log.debug(baseScore)

            for param, name, esti in [(param, name, w) for w in W for name, param in w.parameters.items()
                                      if param.fitStep is not None and len(param.fitStep) > 1]:
                scores = []
                for value in param.fitStep:
                    param.value = value
                    Y_2 = Estimator.forwardPass(X, W)
                    score2 = evalFunction(Y_2, Y)
                    scores.append((score2, value))

                bestIdx = np.argmax([s[0][targetFunction] for s in scores])
                param.value = scores[bestIdx][1]
                log.debug(scores[bestIdx], esti)

        return baseScore
