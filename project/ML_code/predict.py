import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import sys



if len(sys.argv) > 1 :
    mn = sys.argv[1]
    public_fn = sys.argv[2]
    private_fn = sys.argv[3]



def data_preprocess(x):
    x_train = list(x)

    for i, row in enumerate(x_train):
        x_train[i] = list(row)
    for i, row in enumerate(x_train):
        if row[2] not in (1, 2, 3):
            x_train[i][2] = 4

    for row in x_train:
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



    return np.array(x_train)

for _ in range(2) :
    fn = public_fn if _ == 0 else private_fn
    test_data = pd.read_csv(fn)
    test_data = np.array(test_data)[:, 1:]

    test_data = data_preprocess(test_data)

    forest = joblib.load(mn)

    pred_prob = forest.predict_proba(test_data)
    pred_prob = list(pred_prob)
    pred_list = []
    index_start = 1 if _ == 0 else 5001
    for i, pair in enumerate(pred_prob, index_start) :
        pred_list.append([i, pair[1]])

    pred_list = sorted(pred_list, key=lambda x: x[1], reverse=True)
    output = list(np.array(pred_list)[:, 0])
    for i, item in enumerate(output) :
        output[i] = int(item)

    pred_df = pd.DataFrame(output)
    predict_fn = "public" if _ == 0 else "private"
    pred_df.to_csv('%s.csv' % predict_fn, index=False, header=["Rank_ID"])

