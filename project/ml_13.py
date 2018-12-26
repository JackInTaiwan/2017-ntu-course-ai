import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib
import itertools
import sys


def data_preprocess(x):
    x_train = list(x)

    for i, row in enumerate(x_train):
        x_train[i] = list(row)
    for i, row in enumerate(x_train):
        if row[2] not in (1, 2, 3):
            x_train[i][2] = 4

    for row in x_train:
        # Bill
        row.insert(11, np.mean(row[11:16]))
        row.insert(11, np.var(row[12:17]))

        # Pay
        row.insert(5, np.mean(row[5:11]))
        row.insert(5, np.var(row[6:12]))

        # Education
        old_edu = row.pop(2)
        if old_edu == 1:
            row.insert(2, 1)
            row.insert(3, 0)
            row.insert(4, 0)
            row.insert(5, 0)
        elif old_edu == 2:
            row.insert(2, 0)
            row.insert(3, 1)
            row.insert(4, 0)
            row.insert(5, 0)
        elif old_edu == 3:
            row.insert(2, 0)
            row.insert(3, 0)
            row.insert(4, 1)
            row.insert(5, 0)
        else:
            row.insert(2, 0)
            row.insert(3, 0)
            row.insert(4, 0)
            row.insert(5, 1)

        # Age
        old_age = row.pop(4)
        if old_age <= 21:
            row.insert(7, 1)
            row.insert(8, 0)
            row.insert(9, 0)
            row.insert(10, 0)
            row.insert(11, 0)
        elif 21 < old_age <= 31:
            row.insert(7, 0)
            row.insert(8, 1)
            row.insert(9, 0)
            row.insert(10, 0)
            row.insert(11, 0)
        elif 31 < old_age <= 41:
            row.insert(7, 0)
            row.insert(8, 0)
            row.insert(9, 1)
            row.insert(10, 0)
            row.insert(11, 0)
        elif 41 < old_age <= 51:
            row.insert(7, 0)
            row.insert(8, 0)
            row.insert(9, 0)
            row.insert(10, 1)
            row.insert(11, 0)
        else:
            row.insert(7, 0)
            row.insert(8, 0)
            row.insert(9, 0)
            row.insert(10, 0)
            row.insert(11, 1)


        """
        for _ in range(6) :
            pay.append(row.pop(5))

        row.insert(5, np.mean(pay))
        row.insert(6, np.var(pay))
        """
    """
    for i, y in enumerate(y_train[:]):
        if y == 1:
            y_train.append(1)
            x_train.append(x_train[i])
    train_data = list(zip(x_train, y_train))
    random.shuffle(train_data)
    x_train, y_train = list(zip(*train_data))
    """
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    for row in x_train:
        row[0] = row[0] * 5
    return np.array(x_train)


mn = sys.argv[0]
mn = mn.split("/")
mn = mn[-1][0: -3]
fn = sys.argv[1]

### Datac Load
train_data = pd.read_csv("Train.csv")
train_data = np.array(train_data)
x_train = train_data[:, 1:train_data.shape[1] - 1]
y_train_all = train_data[:, -1]
x_train_all = data_preprocess(x_train)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
for train_index, test_index in sss.split(x_train_all, y_train_all):
    x_test, y_test = x_train_all[test_index], y_train_all[test_index]
    x_train, y_train = x_train_all[train_index], y_train_all[train_index]
"""
### Augmentation
x_train = list(x_train)
y_train = list(y_train)
for i, row in enumerate(x_train):
    x_train[i] = list(row)
for i, y in enumerate(y_train[:]):
    if y == 1:
        y_train.append(1)
        x_train.append(x_train[i])
x_train, y_train = np.array(x_train), np.array(y_train)
"""
print(len(y_test[y_test == 1]))

""" NeuralNetwork Classification """


def neuralNetworkClaasification(x_train, y_train, neurons, layers, iteration):
    """
    ### Standardization
    ss = StandardScaler()
    x_train_std = ss.fit_transform(x_train)
    x_test_std = ss.fit_transform(x_test)

    ### Pipe of MLPClassification
    pipe_mlps = Pipeline([
        ('mlps', MLPC(hidden_layer_sizes=(neurons, layers), activation='relu',
                      solver='sgd', batch_size='auto',
                      learning_rate='adaptive', learning_rate_init=0.001,
                      max_iter=iteration, shuffle=True))
    ])

    ### Param range for `alpha`
    param_range = [10 ** i for i in range(-5, 0)]
    train_scores, test_scores = validation_curve(
        estimator=pipe_mlps,
        X=x_train_std,
        y=y_train,
        param_name='mlps__alpha',
        param_range=param_range,
        cv=5
    )

    param_alpha = param_range[np.argmax(test_scores.sum(axis=1))]
    print ("Best alpha: %s" % param_alpha)
    print ("Val scores: %s" % test_scores)

    """
    ### MLP Classification
    mlpc_fin = MLPC(hidden_layer_sizes=(neurons, layers), activation='relu',
                    solver='sgd', alpha=0.001, batch_size=50,
                    learning_rate='adaptive', learning_rate_init=0.01,
                    max_iter=iteration, shuffle=True)
    mlpc_fin.fit(x_train, y_train)

    for i in range(10):

        mlpc_fin.fit(x_train, y_train)
        score_test = mlpc_fin.score(x_test, y_test)
        score = mlpc_fin.score(x_train, y_train)
        pred_prob = mlpc_fin.predict_proba(x_test)
        pred_pair = []
        for j, pair in enumerate(pred_prob):
            pred_pair.append((j, pair[1]))
        pred_pair = sorted(pred_pair, key=lambda x: x[1], reverse=True)
        num = 800
        count = 0
        for pair in pred_pair[0:num]:
            if y_test[pair[0]] == 1: count += 1

        print("____Iteration: %s" % (i * iteration))
        print("Precision: %s" % (count / num))
        print("Score for testing: %s" % score_test)
        print("Score: %s" % score)

    """
    joblib.dump(mlpc_fin, "nn_model_1.pkl")
    score = mlpc_fin.score(x_train_all, y_train_all)
    print ("score: %s" % score)
    """
    f.write("Score for testing: %s \n" % score_test)
    f.write("Score: %s \n" % score)


layers = [3, 6, 10, 15]
neurons = [50, 100, 150, 200]
iteration = 500
print("***Model name: %s" % mn)
f = open(fn, "a+")
f.write("***Model name: %s \n" % mn)
f.close()
for pair in itertools.product(neurons, layers):
    f = open(fn, "a+")
    print("Start new setting : %s %s" % (pair[0], pair[1]))
    f.write("Start new setting : %s %s \n" % (pair[0], pair[1]))
    neuralNetworkClaasification(x_train, y_train, pair[0], pair[1], 500)
    print("\n")
    f.write("\n")
    f.close()