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

POSSIBLE_COLUMNS = ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'HR',
                    'pNN25', 'pNN50', 'SD1', 'SD2', 'KURT', 'SKEW', 'MEAN_REL_RR',
                    'MEDIAN_REL_RR', 'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR',
                    'SDRR_RMSSD_REL_RR', 'KURT_REL_RR', 'SKEW_REL_RR', 'VLF', 'VLF_PCT',
                    'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU', 'TP', 'LF_HF',
                    'HF_LF', 'sampen', 'higuci', 'MEAN_RR_LOG',
                    'MEAN_RR_SQRT', 'TP_SQRT', 'MEDIAN_REL_RR_LOG', 'RMSSD_REL_RR_LOG',
                    'SDSD_REL_RR_LOG', 'VLF_LOG', 'LF_LOG', 'HF_LOG', 'TP_LOG', 'LF_HF_LOG',
                    'RMSSD_LOG', 'SDRR_RMSSD_LOG', 'pNN25_LOG', 'pNN50_LOG', 'SD1_LOG',
                    'KURT_YEO_JONSON', 'SKEW_YEO_JONSON', 'MEAN_REL_RR_YEO_JONSON',
                    'SKEW_REL_RR_YEO_JONSON', 'LF_BOXCOX', 'HF_BOXCOX', 'SD1_BOXCOX',
                    'KURT_SQUARE', 'HR_SQRT', 'MEAN_RR_MEAN_MEAN_REL_RR', 'SD2_LF', 'HR_LF',
                    'HR_HF', 'HF_VLF']

COLUMNS_TO_KEEP = ['MEAN_RR', 'MEDIAN_RR', 'SDRR',
                   'RMSSD', 'RMSSD_LOG', 'SDSD', 'SDRR_RMSSD', 'pNN25', 'pNN50',
                   'HR']

def rf_grid():
    return {'n_estimators': [100, 250, 500], 'max_depth': [2, 3]}

def rf_for_subject_subset(df, subject_threshold):
   included_df = df[df['subject_id'] <= subject_threshold]
   X = included_df
   X = X[COLUMNS_TO_KEEP]
   X = StandardScaler().fit_transform(X)
   y = included_df['NasaTLX Label'].values
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   clf = GridSearchCV(rf, rf_grid())
   clf.fit(X_train, y_train)
   print(f'Accuracy Score for Included Subset: {accuracy_score(y_test, clf.predict(X_test))}')
   return clf

def test_rf_on_excluded_subset(clf, subject_threshold):
    excluded_df = df[df['subject_id'] > subject_threshold]
    X_test = excluded_df[COLUMNS_TO_KEEP]
    X_test = StandardScaler().fit_transform(X_test)
    y_test = excluded_df['NasaTLX Label'].values
    print('Accuracy Score for Excluded Subset')
    return accuracy_score(y_test, clf.predict(X_test))

def calibrate_rf_with_sample_of_excluded_subset(df, subject_threshold, samples_per_subject):
    subset_df = df[df['subject_id'] <= subject_threshold]
    excluded_df = df[df['subject_id'] > subject_threshold]
    copy_of_excluded_df = excluded_df.copy()
    random_samples_idxs = []

    for i in copy_of_excluded_df['subject_id'].unique():
        random_sample = df[df['subject_id'] == i].sample(n=samples_per_subject)
        random_samples_idxs.append(random_sample.index)
        subset_df = pd.concat([subset_df, random_sample])

    calibrated_rf = rf_for_subject_subset(subset_df, subject_threshold)
    # remove random sample from excluded df for test
    print(f'TOTAL IDXS: {len(np.array(random_samples_idxs).flatten())}')
    excluded_df = excluded_df.drop(np.array(random_samples_idxs).flatten())

    X_test = excluded_df[COLUMNS_TO_KEEP]
    X_test = StandardScaler().fit_transform(X_test)
    y_test = excluded_df['NasaTLX Label'].values
    print(
        f'Accuracy Score for Excluded Subset: {accuracy_score(y_test, calibrated_rf.predict(X_test))}')
    return calibrated_rf
