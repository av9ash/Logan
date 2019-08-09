from sklearn.datasets import load_files
import joblib
import datetime
import os
from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
from pandas import DataFrame
import matplotlib.pyplot as plt

now = datetime.datetime.now()
time_stamp = now.strftime("%Y_%b_%d_%H_%M")
print('Testing Stamp:'+time_stamp)


def add_to_log(line):
    line = str(line)
    with open('execution_log.txt', 'a') as log:
        log.write(line)
        log.write('\n')
        if line == 'Done':
            log.write('-'*50)
            log.write('\n')

    print(line)


def load_file(file_name):
    obj_file = joblib.load(file_name)
    add_to_log('File loaded ' + file_name)
    return obj_file


def get_testing_data(path):
    reviews_test = load_files(path)
    text_test, y_test = reviews_test.data, reviews_test.target
    add_to_log('Got Testing Data '+path)
    return text_test, y_test


def score_accuracy(X_test, y_test, clf):
    add_to_log('Scoring Model..')
    try:
        y_preds = clf.predict(X_test)
    except TypeError as e:
        print(e)
        add_to_log('Using Dense array')
        y_preds = clf.predict(X_test.toarray())

    print_confusion_matrix(y_test, y_preds, clf)


def score_voting_accuracy(X_test, y_test, clf):
    add_to_log('Scoring Model..')
    rfc, gnb = clf
    y_preds = v_predict(X_test, rfc, gnb)

    acc = np.mean(y_preds == y_test)
    add_to_log('accurary: ' + str(acc))

    print_confusion_matrix(y_test, y_preds, clf)


def v_predict(X_test, rfc, gnb):

    add_to_log('Predicting RFC')
    cond_prob = rfc.predict_proba(X_test)

    add_to_log('Predicting GNB')
    gnb_cond_prob = gnb.predict_proba(X_test.toarray())

    add_to_log('Voting')
    for i, row in enumerate(cond_prob):
        counter = Counter(row)
        del (counter[0.0])
        del (counter[1.0])
        if len(counter) == 1:
            cond_prob[i] = gnb_cond_prob[i]

    add_to_log('Prediction Done')
    y_preds = np.argmax(cond_prob, axis=1)
    return y_preds


def save_test_matrix(matrix, reduced=False):

    if reduced:
        file_path = os.path.join('matrix', 'test_'+time_stamp+'_mtrx_reduced.joblib')
    else:
        file_path = os.path.join('matrix', 'test_'+time_stamp+'_mtrx.joblib')

    joblib.dump(matrix, file_path)
    add_to_log('Matrix Saved '+ file_path)


def save_test_vector(vector):

    file_path = os.path.join('vectors', 'test_'+time_stamp+'_vctr.joblib')
    joblib.dump(vector, file_path)

    add_to_log('Vector Saved')


def selector_transform(selectors, X_test):
    add_to_log('Applying Selectors')

    for selector in selectors:
        X_test = selector.transform(X_test)

    add_to_log('Features Transformed. '+str(X_test.shape[1]))
    return X_test


def print_confusion_matrix(y_expec, y_preds, clf):


    clf_type = str(type(clf))
    clf_name = clf_type.split("'")[1].split('.')[-1]

    file_path = os.path.join('cnf_mtrx', time_stamp+clf_name)

    conf_mat = confusion_matrix(y_true=y_expec, y_pred=y_preds)
    conf_mat_pr = []
    for row in conf_mat:
        conf_mat_pr.append((row / sum(row)))

    add_to_log(conf_mat)
    acc = np.mean(y_preds == y_expec)
    add_to_log('accurary: '+str(acc))

    # The order of labels is important
    labels = ['Hardware', 'Other', 'Script', 'Software', 'Tools']
    df_cm = DataFrame(conf_mat, index=labels, columns=labels)
    df_prec = DataFrame(conf_mat_pr, index=labels, columns=labels)

    sns_plot = sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    sns_plot.set_title("Acc: " + str(acc))
    plt.savefig(file_path)
    plt.figure()

    sns_plot = sn.heatmap(df_prec, annot=True, cmap='Blues', fmt='.2%')
    sns_plot.set_title("Acc: " + str(acc))
    plt.savefig(file_path+'_pr')
    plt.figure()


def test_model(vect_file, model_file, data_paths, selectors=None, summary=None, X_test=None):
    for data_path in data_paths:
        vect = load_file(vect_file)
        print(list(selectors))
        text_test, y_test = get_testing_data(data_path)

        if X_test is not None and os.path.exists(X_test):
            print('Loading Test Matrix')
            X_test = load_file(X_test)
        else:
            print('Transforming Test Matrix')
            X_test = vect.transform(text_test)
            save_test_matrix(X_test)

        if selectors is not None:
            print('Transforming Features')
            X_test = selector_transform(selectors, X_test)

        if summary is not None:
            add_to_log(summary)

        clf = load_file(model_file)

        score_accuracy(X_test, y_test, clf)
        add_to_log('Done')


def main():
    vect_file = 'vectors/2019_Jul_16_12_24_vctr.joblib'
    model_file = 'models/2019_Jul_18_12_26_clfr.joblib'
    data_paths = [os.path.join(os.getcwd(), 'june_data')]
    selectors = load_file('selector/2019_Jul_18_12_26_selector.joblib')

    X_test = None
    X_test = 'matrix/test_2019_Jul_16_12_24_mtrx.joblib'

    test_model(vect_file,model_file,data_paths,selectors,None, X_test)


if __name__ == '__main__':
    main()
