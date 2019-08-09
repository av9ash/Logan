from sklearn.datasets import load_files
import joblib
import csv
from collections import OrderedDict
from ML_Model import Model
import numpy as np


def load_file(file_name):
    obj_file = joblib.load(file_name)
    return obj_file


def selector_transform(selectors, X_test):
    for selector in selectors:
        X_test = selector.transform(X_test)
    return X_test


def main():
    print('Main')
    timestamp = '2019_Jul_31_15_55'
    dirt = '/Users/patila/Desktop/'

    # tm_path = dirt+'Trained_Model/rfc_gini/'
    # vect = load_file(tm_path+timestamp+'_vctr.joblib')
    #
    # timestamp = '2019_Jul_17_17_07'
    # selectors = load_file(tm_path+timestamp+'_selector.joblib')
    # clf = load_file(tm_path+timestamp+'_clfr.joblib')

    model = Model()
    model.load_model(timestamp)

    labels = ['Hardware', 'Other', 'Script', 'Software', 'Tools']

    dir_path = dirt+'june'

    src_csv = dirt+'June_Data.csv'

    logs_test = load_files(dir_path)

    # Has to be an ordered dictionary
    file_names = OrderedDict((file_name.replace(dir_path+'/unlabeled/', ''), i)
                             for (i, file_name) in enumerate(logs_test.filenames))

    text_test = logs_test.data
    print('Files loaded:', len(text_test))

    print('Applying Vector')
    X_test = model.fit_transform(text_test)

    print('Applying Classfier')
    cond_prob = model.clf.predict_proba(X_test)
    y_pred = model.clf.predict(X_test)

    print(len(cond_prob), len(y_pred))

    with open(src_csv, encoding='utf-8-sig') as csv_file:
        with open('june_predictions.csv', 'w') as dwn_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                exec_id = row['test_exec_id']+'.txt'
                actual_label = row['debug_tag']

                if exec_id in file_names:
                    idx = file_names[exec_id]
                    outrow = [exec_id, actual_label, labels[y_pred[idx]], cond_prob[idx]]
                    t_writer = csv.writer(dwn_file)
                    t_writer.writerow(outrow)

    print('Done')


if __name__ == '__main__':
    main()
