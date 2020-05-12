import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, GridSearchCV

COLUMNS_TO_KEEP = ['MEAN_RR', 'MEDIAN_RR', 'SDRR',
                   'RMSSD', 'SDSD', 'SDRR_RMSSD', 'pNN25', 'pNN50',
                   'HR']

def rf_grid():
    return {'n_estimators': [100, 250, 500], 'max_depth': [2, 3]}

def rf_for_subject_subset(df, subject_threshold):
   X = df[df['subject_id'] <= subject_threshold]
   X = X[COLUMNS_TO_KEEP]
   X = StandardScaler().fit_transform(X)
   y = df[df['subject_id'] <= subject_threshold]['NasaTLX Label'].values
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   clf = GridSearchCV(rf, rf_grid())
   clf.fit(X_train, y_train)
   print(f'Accuracy Score: {accuracy_score(y_test, clf.predict(X_test))}')
   return clf

def test_rf_on_excluded_subset(clf, subject_threshold):
    X_test = df[df['subject_id'] > subject_threshold]
    y_test = df[df['subject_id'] > subject_threshold]['NasaTLX Label'].values
    return accuracy_score(y_test, clf.predict(X_test))

def calibrate_rf_with_sample_of_excluded_subset(df, subject_threshold):
    subset_df = df[df['subject_id'] <= subject_threshold]
    random_sample = df[df['subject_id'] > subject_threshold].sample(n=1000)
    new_df = pd.concat([subset_df, random_sample])
    return rf_for_subject_subset(new_df, subject_threshold)
