from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
import numpy as np
import json

from RF_Main_Process import RF_Main_Process

if __name__ == "__main__":
    digits = load_digits()       

    is_rf= True
    add_reduce = False

    robust_epsilon = 1

    num_estimators = 5
    num_max_depth = 3
    model_name = 'rf_c_digits_%s_%s' % (num_estimators, num_max_depth)

    clf = RandomForestClassifier(n_estimators=num_estimators, max_depth=num_max_depth)
    clf.fit(digits.data, digits.target)
    score_pre = cross_val_score(clf, digits.data, digits.target, cv=10).mean()
    print("Accuracy:%s" % score_pre.mean())

    main = RF_Main_Process(model_name, is_rf, add_reduce, robust_epsilon, clf, digits.data, digits.target)

    main.process_digits()




