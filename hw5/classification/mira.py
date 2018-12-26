# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
import random
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.

        ## Something you might need to use:
        ## trainingData[i]: a feature vector, an util.Counter()
        ## trainingLabels[i]: label for each trainingData[i]
        ## self.weights[label]: weight vector for a label (class), an util.Counter()
        ## self.classify(data): this method might be needed in validation
        ## Cgrid: a list of constant C
        """
        "*** YOUR CODE HERE ***"
        val_accs = []
        weights_list = []

        for c in Cgrid :
            weights_iter = self.weights.copy()
            for _ in range(self.max_iterations) :
                for i in range(len(trainingData)):
                    max_value = -float('inf')
                    pred_label = None
                    for label in self.legalLabels:
                        if weights_iter[label] == {}:
                            for f in self.features:
                                weights_iter[label][f] = 0.01
                        value = trainingData[i] * weights_iter[label]
                        if value > max_value:
                            max_value = value
                            pred_label = label
                        elif value == max_value and pred_label == trainingLabels[i]:
                            pred_label = label

                    if pred_label != trainingLabels[i]:
                        weights = weights_iter.copy()
                        tor = min([ c, (((weights[pred_label] - weights[label]) * trainingData[i]) + 2) / (2 * ( trainingData[i] * trainingData[i]) ** 0.5)])
                        delta = util.Counter()
                        for f in trainingData[i] :
                            delta[f] = trainingData[i][f] * tor
                        weights_iter[trainingLabels[i]] += delta
                        weights_iter[pred_label] -= delta
                pairs = list(zip(trainingData, trainingLabels))
                random.shuffle(pairs)
                trainingData, trainingLabels = list(zip(*pairs))
            # Validation
            correct = 0
            for i in range(len(validationData)) :
                pred_label, max_value = None, -float('inf')
                for label in self.legalLabels :
                    value = weights_iter[label] * validationData[i]
                    if value > max_value :
                        pred_label, max_value = label, value
                if pred_label == validationLabels[i] : correct += 1
            val_accs.append(correct/float(len(validationData)))
            weights_list.append(weights_iter)
        print (val_accs)
        max_acc = 0
        fin_weights = None
        fin_c = None
        for i, acc in enumerate(val_accs) :
            if acc > max_acc or (acc == max_acc and Cgrid[i] < fin_c):
                max_acc, fin_weights, fin_c = acc, weights_list[i], Cgrid[i]
        self.weights = fin_weights


    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


