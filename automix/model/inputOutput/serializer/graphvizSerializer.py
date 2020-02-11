import itertools
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph

from automix.featureExtraction import Estimator
from automix.model.classes import Signal
from automix.model.inputOutput.serializer.serializer import Serializer


class GraphvizSerializer(Serializer):
    """
    class handling jams serialization of track or mixes
    """

    def addslashes(self, s):
        forbiddenCharacter = ["\\", "\0", "'", '"']
        for i in forbiddenCharacter:
            if i in s:
                s = s.replace(i, "")
        return s

    def serializeEstimators(self, graph: List[Estimator], filename="computation_graph.pdf", features=None):

        s = Digraph('structs', node_attr={'shape': 'plaintext'}, filename=filename)
        # Add nodes with inputs and outputs
        tmpFiles = []
        for i, estimator in enumerate(graph):
            # s.attr('node', fillcolor='yellow', style='filled')
            inputs = "<TR>" + "".join(["<TD PORT='" + el + "'>" + el + "</TD>" for el in np.hstack(estimator.inputs)]) + "</TR>"
            name = "<TR><TD COLSPAN='" + str(max(len(np.hstack(estimator.inputs)), len(
                estimator.outputs))) + "'>" + self.addslashes(str(estimator)) + "</TD></TR>"
            outputs = "<TR>" + "".join(["<TD PORT='" + el + "'>" + el + "</TD>" for el in estimator.outputs]) + "</TR>"

            # Save temporary images to put on the graph
            imageFile = None
            if features:
                imageFile = str(i) + "tmp.png"
                tmpFiles.append(imageFile)
                figure = plt.figure()
                for output in [o for o in estimator.outputs if o in features]:
                    if type(features[output]) == Signal:
                        features[output].plot(label=output)
                if len(estimator.outputs) > 1:
                    plt.legend()
                figure.savefig(imageFile)
                plt.close(figure)

            s.node(str(i),
                   label='<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">' + inputs + name + outputs + '</TABLE>>',
                   image=imageFile,
                   imagepos='ml',
                   imagescale='true')

        # Add edges
        edges = []
        for i, source in enumerate(graph):
            for j, target in enumerate(graph[i + 1:]):
                j = j + i + 1
                for output, input in itertools.product(source.outputs, np.hstack(target.inputs)):
                    if input == output:
                        edges.append((str(i) + ":" + input, str(j) + ":" + output))
        s.edges(edges)
        s.view()

        for f in tmpFiles:
            os.remove(f)
