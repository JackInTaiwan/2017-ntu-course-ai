import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit
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
        row.insert(11, 5 * np.mean(row[11:17]))
        row.insert(11, np.var(row[12:18]))

        # Pay
        row.insert(5, 5 * np.mean(row[5:11]))
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
        row.pop(6)
        """
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
    """
    for i, y in enumerate(y_train[:]):
        if y == 1:
            y_train.append(1)
            x_train.append(x_train[i])
    train_data = list(zip(x_train, y_train))
    random.shuffle(train_data)
    x_train, y_train = list(zip(*train_data))
    """

    return np.array(x_train)


def test_score(forest_model, x_test, y_test, num = 400):
    pred_prob = forest_model.predict_proba(x_test)
    pred_pair = []
    for j, pair in enumerate(pred_prob):
        pred_pair.append((j, pair[1]))
    pred_pair = sorted(pred_pair, key=lambda x: x[1], reverse=True)
    count = 0
    for pair in pred_pair[0:num]:
        if y_test[pair[0]] == 1: count += 1

    return count / num



mn = sys.argv[0]
mn = mn.split("/")
mn = mn[-1][0: -3]
print(mn)

for i in range(1, 4):
    ### Datac Load
    train_data = pd.read_csv("Train.csv")
    train_data = np.array(train_data)
    x_train = train_data[:, 1:train_data.shape[1] - 1]
    y_train_all = train_data[:, -1]
    x_train_all = data_preprocess(x_train)

    ### Split Testing Data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
    for train_index, test_index in sss.split(x_train_all, y_train_all):
        x_test, y_test = x_train_all[test_index], y_train_all[test_index]
        x_train, y_train = x_train_all[train_index], y_train_all[train_index]

    train_size = x_train.shape[0]
    k = 10
    n_trees = 30
    class_w = {0: 1, 1: 1}

    ### Pipe 1 of decision tree
    param_range = range(5, 23, 3)
    param_scores = []
    for param in param_range :
        scores = []
        sss_cross = StratifiedShuffleSplit(n_splits=k, test_size=0.1, random_state=1)
        for train_index, test_index in sss_cross.split(x_train, y_train):
            forest = RandomForestClassifier(criterion='entropy',
                                            n_estimators=n_trees, n_jobs=4,
                                            min_samples_split=50,
                                            min_impurity_decrease=0.0002,
                                            class_weight=class_w,
                                            max_depth=param)
            x_t, y_t = x_train[test_index], y_train[test_index]
            x_tr, y_tr = x_train[train_index], y_train[train_index]
            forest.fit(x_tr, y_tr)
            k_cross_score = test_score(forest, x_t, y_t, num = 300)
            scores.append(k_cross_score)
        param_scores.append(np.mean(scores))

    print (param_scores)
    max_depth_best = param_range[np.argmax(param_scores)]
    print(max_depth_best)


    ### Pipe 2 of decision tree
    param_range = range(50, 200, 20)
    param_scores = []
    for param in param_range:
        scores = []
        sss_cross = StratifiedShuffleSplit(n_splits=k, test_size=0.1, random_state=1)
        for train_index, test_index in sss_cross.split(x_train, y_train):
            forest = RandomForestClassifier(criterion='entropy',
                                            n_estimators=n_trees, n_jobs=4,
                                            min_samples_split=param,
                                            min_impurity_decrease=0.0002,
                                            class_weight=class_w,
                                            max_depth=max_depth_best)
            x_t, y_t = x_train[test_index], y_train[test_index]
            x_tr, y_tr = x_train[train_index], y_train[train_index]
            forest.fit(x_tr, y_tr)
            k_cross_score = test_score(forest, x_t, y_t, num=300)
            scores.append(k_cross_score)
        param_scores.append(np.mean(scores))

    print (param_scores)
    min_samples_best = param_range[np.argmax(param_scores)]
    print(min_samples_best)


    ### Final tree with the params above
    forest_fin = RandomForestClassifier(
        criterion='entropy',
        n_jobs=4,
        min_samples_split=int(min_samples_best * k / (k - 1)),
        n_estimators=n_trees,
        max_depth=int(max_depth_best * k / (k - 1)),
        min_impurity_decrease=0.0002,
        class_weight=class_w
    )
    forest_fin.fit(x_train, y_train)
    score_test = test_score(forest_fin, x_test, y_test)
    score = forest_fin.score(x_train, y_train)
    print("___" + str(i))
    print("Testing score: %s" % score_test)
    print("Training score: %s" % score)

    ### Final tree on all training data
    forest_fin_all = RandomForestClassifier(
        criterion='entropy',
        n_jobs=4,
        min_samples_split=int(min_samples_best * k / (k - 1) * (10 / 9)),
        n_estimators=n_trees,
        max_depth=int(max_depth_best * k / (k - 1)),
        min_impurity_decrease=0.0002,
        class_weight=class_w
    )
    forest_fin_all.fit(x_train_all, y_train_all)
    score_all = forest_fin_all.score(x_train_all, y_train_all)
    print("All score: %s" % score_all)

    f = open("record_37.txt", "a+")
    f.write("Model name: %s \n" % mn)
    f.write("Iteration : %s \n" % i)
    f.write("Parameters: %s %s %s \n" % (max_depth_best, 0.0002, min_samples_best))
    f.write("Testing Score: %s \n" % str(score_test))
    f.write("Training Score: %s \n" % str(score))
    f.write("Score: %s \n \n" % score_all)
    f.close()

    joblib.dump(forest_fin, "%s_%s.pkl" % (mn, i))



