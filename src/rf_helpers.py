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

POSSIBLE_COLUMNS = ['HF',
                    'HF_BOXCOX',
                    'HF_LF',
                    'HF_LOG',
                    'HF_NU',
                    'HF_PCT',
                    'HF_VLF',
                    'HR',
                    'HR_HF',
                    'HR_LF',
                    'HR_SQRT',
                    'KURT',
                    'KURT_REL_RR',
                    'KURT_SQUARE',
                    'KURT_YEO_JONSON',
                    'LF',
                    'LF_BOXCOX',
                    'LF_HF',
                    'LF_HF_LOG',
                    'LF_LOG',
                    'LF_NU',
                    'LF_PCT',
                    'MEAN_REL_RR',
                    'MEAN_REL_RR_YEO_JONSON',
                    'MEAN_RR',
                    'MEAN_RR_LOG',
                    'MEAN_RR_MEAN_MEAN_REL_RR',
                    'MEAN_RR_SQRT',
                    'MEDIAN_REL_RR',
                    'MEDIAN_REL_RR_LOG',
                    'MEDIAN_RR',
                    'RMSSD',
                    'RMSSD_LOG',
                    'RMSSD_SQUARED',
                    'RMSSD_REL_RR',
                    'RMSSD_REL_RR_LOG',
                    'SD1',
                    'SD1_BOXCOX',
                    'SD1_LOG',
                    'SD2',
                    'SD2_LF',
                    'SDRR',
                    'SDRR_REL_RR',
                    'SDRR_RMSSD',
                    'SDRR_RMSSD_LOG',
                    'SDRR_RMSSD_REL_RR',
                    'SDSD',
                    'SDSD_REL_RR',
                    'SDSD_REL_RR_LOG',
                    'SKEW',
                    'SKEW_REL_RR',
                    'SKEW_REL_RR_YEO_JONSON',
                    'SKEW_YEO_JONSON',
                    'TP',
                    'TP_LOG',
                    'TP_SQRT',
                    'VLF',
                    'VLF_LOG',
                    'VLF_PCT',
                    'higuci',
                    'pNN25',
                    'pNN25_LOG',
                    'pNN50',
                    'pNN50_LOG',
                    'sampen',
                    'NasaTLX Label']

COLUMNS_TO_KEEP = ['HF',
                   'HF_BOXCOX',
                   'HF_LF',
                   'HF_NU',
                   'HF_PCT',
                   'HR',
                   'HR_HF',
                   'HR_LF',
                   'HR_SQRT',
                   'LF',
                   'LF_BOXCOX',
                   'LF_HF',
                   'LF_NU',
                   'LF_PCT',
                   'MEAN_RR',
                   'MEDIAN_RR',
                   'RMSSD',
                   'RMSSD_LOG',
                   'RMSSD_SQUARED',
                   'RMSSD_REL_RR',
                   'SD1_BOXCOX',
                   'SDRR',
                   'SDRR_RMSSD',
                   'SDRR_RMSSD_REL_RR',
                   'SDSD',
                   'TP',
                   'VLF',
                   'VLF_PCT',
                   'pNN25',
                   'pNN50']

RF_GRID = {'n_estimators': [100, 250, 500, 1000], 'max_depth': [2, 3]}

def random_included_and_excluded_df(df, total_excluded=5):
    random_subject_ids = np.random.choice(df['subject_id'].unique(), total_excluded)
    excluded_df = df[df['subject_id'].isin(
        random_subject_ids)][POSSIBLE_COLUMNS]
    included_df = df[~df['subject_id'].isin(
        random_subject_ids)][POSSIBLE_COLUMNS]
    return included_df, excluded_df

# NasaTLX label is low, medium, high which is the mapping done in the study
def rf_for_subject_subset(included_df):
   X = included_df.drop(columns='NasaTLX Label')
   X = StandardScaler().fit_transform(X)
   y = included_df['NasaTLX Label'].values
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   clf = GridSearchCV(RandomForestClassifier(), RF_GRID)
   clf.fit(X_train, y_train)
   print(f'Accuracy Score for Included Subset: {accuracy_score(y_test, clf.predict(X_test))}')
   return clf

def test_rf_on_excluded_subset(clf, excluded_df):
    X_test = excluded_df.drop(columns='NasaTLX Label')
    X_test = StandardScaler().fit_transform(X_test)
    y_test = excluded_df['NasaTLX Label'].values
    print(f'Accuracy Score for Excluded Subset without Calibration: {accuracy_score(y_test, clf.predict(X_test))}')

def combined_included_excluded_without_calibration(included_df, excluded_df):
    clf = rf_for_subject_subset(included_df)
    test_rf_on_excluded_subset(clf, excluded_df)

# Default sample set to around 6.25% of data
# This is not a great method to calibrate given sampled datapoints will be correlated with excluded datapoints
# Ideally this should be done with fresh data points through a cycle of scenarios early in the prediction process
def calibrated_rf_with_sample_of_excluded_subset(included_df, excluded_df, samples_per_subject=1000):
    for i in excluded_df['subject_id'].unique():
        random_sample = excluded_df[excluded_df['subject_id'] == i].sample(n=samples_per_subject)
        included_df = pd.concat([included_df, random_sample])
        # remove random sample from excluded subset
        excluded_df.drop(random_sample.index, inplace=True)

    calibrated_rf = rf_for_subject_subset(included_df)
    X_test = excluded_df.drop(columns='NasaTLX Label')
    X_test = StandardScaler().fit_transform(X_test)
    y_test = excluded_df['NasaTLX Label'].values
    print(
        f'Accuracy Score for Excluded Subset with Calibrated RF: {accuracy_score(y_test, calibrated_rf.predict(X_test))}')
    return calibrated_rf

def test_on_validation_df(validation_df):
    X_test = validation_df[POSSIBLE_COLUMNS].drop(columns='NasaTLX Label')
    X_test = StandardScaler().fit_transform(X_test)
    y_test = validation_df['NasaTLX Label'].values
    print(f'Accuracy Score for Validation Subset without Calibration: {accuracy_score(y_test, clf.predict(X_test))}')

def rf_predictions_for_each_subject(df):
    predictions = []
    for i in df['subject_id'].unique():
        subject_df = df[df['subject_id'] == i]
        X = subject_df
        X = X[POSSIBLE_COLUMNS]
        X = StandardScaler().fit_transform(X)
        y = subject_df['NasaTLX Label'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clf = GridSearchCV(RandomForestClassifier(), RF_GRID)
        clf.fit(X_train, y_train)
        predictions.append(clf.predict(X_test))
        print(f'Accuracy Score for Included Subject {i}: {accuracy_score(y_test, clf.predict(X_test))}')
    return predictions
