import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import sys



if len(sys.argv) > 1 :
    mn = sys.argv[1]
    fn = sys.argv[2]
test_data = pd.read_csv("Test_Public.csv")
test_data = np.array(test_data)[:, 1:]



def data_preprocess(x):
    x_train = list(x)

    for i, row in enumerate(x_train):
        x_train[i] = list(row)
    for i, row in enumerate(x_train):
        if row[2] not in (1, 2, 3):
            x_train[i][2] = 4

    for row in x_train:
        # Pay-AMT
        row.insert(17, 5 * np.mean(row[17:23]))
        row.insert(17, np.var(row[18:24]))

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


test_data = data_preprocess(test_data)

forest = joblib.load(mn)

pred_prob = forest.predict_proba(test_data)
pred_prob = list(pred_prob)
pred_list = []
for i, pair in enumerate(pred_prob, 1) :
    pred_list.append([i, pair[1]])

pred_list = sorted(pred_list, key=lambda x: x[1], reverse=True)
output = list(np.array(pred_list)[:, 0])
for i, item in enumerate(output) :
    output[i] = int(item)

print (len(output))
pred_df = pd.DataFrame(output)
pred_df.to_csv('%s.csv' % fn, index=False, header=["Rank_ID"])

