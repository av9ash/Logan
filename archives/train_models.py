import joblib
import datetime
import os
from collections import Counter
import test_models
import numpy as np
import time

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, f_classif, chi2

from imblearn.over_sampling import SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

now = datetime.datetime.now()
time_stamp = now.strftime("%Y_%b_%d_%H_%M")

print('Training Stamp:' + time_stamp)
mnb = MultinomialNB(alpha=0.01)
bnb = BernoulliNB()
gnb = GaussianNB()
cnb = ComplementNB()

svc = SGDClassifier(max_iter=1000, tol=1e-3, fit_intercept=True)

lda = LinearDiscriminantAnalysis(solver='svd')

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
nc = NearestCentroid()
rnc = RadiusNeighborsClassifier(n_jobs=-1)

lpg = LabelPropagation(n_jobs=-1)
lps = LabelSpreading(n_jobs=-1)

dct = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', random_state=9)

rfc = RandomForestClassifier(class_weight='balanced', n_jobs=-1, criterion='gini',
                             n_estimators=50, warm_start=True)

etc = ExtraTreesClassifier(n_estimators=1, class_weight='balanced',
                           criterion='entropy', random_state=9, warm_start=True, n_jobs=-1)

ovr = OneVsRestClassifier(LogisticRegression(fit_intercept=True), n_jobs=-1)

gb = GradientBoostingClassifier(n_estimators=100, verbose=1, learning_rate=0.1, random_state=0)
mlp = MLPClassifier()

votc3 = VotingClassifier(estimators=[('rfc', rfc), ('gnb', gnb)], voting='soft', weights=[1, 1])
votc2 = VotingClassifier(estimators=[('rfc', rfc), ('gnb', gnb)], voting='soft', weights=[2, 1])

all_clf = [rfc]

# all_clf = [mnb, gnb, bnb, svc, lda, knn, nc, rnc, dct, rfc, ovr, etc, lpg, lps]

# for i in range(1, 10):
#     all_clf.append(KNeighborsClassifier(n_neighbors=i*10,n_jobs=-1))


def add_to_log(line):
    line = str(line)
    with open('execution_log.txt', 'a') as log:
        log.write(line)
        log.write('\n')
        if line == 'Done':
            log.write('-' * 50)
            log.write('\n')

    print(line)


def load_file(file_name):
    obj_file = joblib.load(file_name)
    add_to_log('File loaded ' + file_name)
    return obj_file


def get_training_data(path):
    logs_train = load_files(path)
    text_train, y_train = logs_train.data, logs_train.target
    add_to_log('Got Training Data')
    print('Classes', np.unique(y_train))
    return text_train, y_train


def save_classifier(classifier):
    file_path = os.path.join('models', now.strftime("%Y_%b_%d_%H_%M") + '_clfr.joblib')
    joblib.dump(classifier, file_path)

    add_to_log('Classifier Saved ' + file_path)

    return file_path


def save_matrix(matrix, reduced=False):
    if reduced:
        file_path = os.path.join('matrix', time_stamp + '_mtrx_reduced.joblib')
    else:
        file_path = os.path.join('matrix', time_stamp + '_mtrx.joblib')

    joblib.dump(matrix, file_path)
    add_to_log('Matrix Saved ' + file_path)


def save_sampled_matrix(matrix, labels):
    file_path_X = os.path.join('synthetic', time_stamp + '_mtrx_syn.joblib')
    file_path_Y = os.path.join('synthetic', time_stamp + '_lbls_syn.joblib')

    joblib.dump(matrix, file_path_X)
    joblib.dump(labels, file_path_Y)
    add_to_log('Matrix Saved: ' + file_path_X + ' ' + file_path_Y)


def save_vector(vector):
    file_path = os.path.join('vectors', time_stamp + '_vctr.joblib')
    joblib.dump(vector, file_path)

    add_to_log('Vector Saved ' + file_path)
    if hasattr(vector, 'n_features'):
        add_to_log(vector.n_features)
    else:
        add_to_log(len(vector.get_feature_names()))

    add_to_log(vector.token_pattern)
    add_to_log(vector.ngram_range)
    if vector.stop_words is not None:
        add_to_log('Total Stop Words: ' + str(len(vector.stop_words)))
    else:
        add_to_log('No Stop Words')

    return file_path


def save_vocab(vector):
    file_path = os.path.join('vocab', time_stamp + '_voca.txt')

    with open(file_path, 'w') as file:
        for n_gram, freq in vector.vocabulary_.items():
            file.write(n_gram + ',' + str(freq) + '\n')


def save_selector(selector):
    file_path = os.path.join('selector', now.strftime("%Y_%b_%d_%H_%M") + '_selector.joblib')
    joblib.dump(selector, file_path)
    add_to_log('Selector Saved ' + file_path)
    return file_path


def train_classifier(X_train, y_train, clf):

    try:
        if hasattr(clf, 'partial_fit'):
            clf.partial_fit(X_train, y_train, np.unique(y_train))
        elif hasattr(clf, 'fit'):
            clf.fit(X_train, y_train)
        else:
            print('Classifier does not have fit methods')

    except TypeError:
        print('Using Dense array')
        if hasattr(clf, 'partial_fit'):
            clf.partial_fit(X_train.toarray(), y_train, np.unique(y_train))
        elif hasattr(clf, 'fit'):
            clf.fit(X_train.toarray(), y_train)
        else:
            print('Classifier does not have fit methods')

    finally:
        add_to_log('Model Trained')
        return save_classifier(clf)


def train_dual_clf(X_train, y_train, clf):
    rfc, gnb = clf

    add_to_log('Training RFC')
    rfc.fit(X_train, y_train)

    add_to_log('Training GNB')
    gnb.partial_fit(X_train.toarray(), y_train, np.unique(y_train))

    t_clf = (rfc, gnb)
    add_to_log('Dual Model Trained')
    return save_classifier(t_clf)


def update_clf(time_stamp):
    c_file = os.path.join('models', time_stamp + '_clfr.joblib')
    if os.path.exists(c_file):
        clf = load_file(c_file)
        clf.n_estimators += 3
        return clf


def over_sample_data(matrix, y_train):
    add_to_log('Over Sampling')
    add_to_log('Sample distribution %s' % Counter(y_train))
    b_line = KMeansSMOTE(k_neighbors=5, sampling_strategy='not majority', n_jobs=-1, random_state=3, kmeans_estimator=100)
    matrix_resampled, y_resampled = b_line.fit_resample(matrix, y_train)
    add_to_log('Resample distribution %s' % Counter(y_resampled))
    return matrix_resampled, y_resampled


def under_sample_data(matrix, y_train):
    add_to_log('Under Sampling')
    add_to_log('Sample distribution %s' % Counter(y_train))
    # clean proximity samples using TomeKLinks
    tl = TomekLinks(random_state=11, sampling_strategy='majority', n_jobs=-1)
    X_res, y_res = tl.fit_resample(matrix, y_train)
    add_to_log('TomekLinks distribution %s' % Counter(y_res))

    enn = EditedNearestNeighbours(random_state=7, sampling_strategy='majority', n_jobs=-1)
    X_res, y_res = enn.fit_resample(X_res, y_res)

    add_to_log('EditedNearestNeighbours distribution %s' % Counter(y_res))
    return X_res, y_res


def feature_selector(X_train, y_train):
    print('Reducing features')
    n_features = X_train.shape[1]
    selectors = []

    # 0.95 * (1 - 0.95) remove feature that have same value in more than 90% of documents
    vth = VarianceThreshold()
    X_train = vth.fit_transform(X_train, y_train)
    add_to_log('0 Variance features: ' + str(n_features - X_train.shape[1]))
    selectors.append(vth)

    k_best = SelectPercentile(chi2, percentile=15)
    X_train = k_best.fit_transform(X_train, y_train)
    selectors.append(k_best)
    add_to_log('KBest-f_reg reduced features to ' + str(X_train.shape[1]))

    return X_train, selectors


def main():
    selectors = None
    X_test = None
    X_train = None
    y_train = None
    # Data Paths
    cwd = '/Users/patila/Desktop'
    global time_stamp

    time_stamp = '2019_Jul_16_12_24'

    train_data_paths = [os.path.join(cwd, 'feb2may_data')]
    test_data_paths = [os.path.join(cwd, 'june_data')]

    for i, data_path in enumerate(train_data_paths):

        v_file = os.path.join('vectors', time_stamp + '_vctr.joblib')
        mtrx = os.path.join('matrix', time_stamp + '_mtrx.joblib')

        # f2m_et
        # time_stamp = '2019_Jul_12_15_27'

        add_to_log(train_data_paths[i] + ' Training')

        text_train, y_train = get_training_data(data_path)

        if os.path.exists(v_file):
            add_to_log('Loading Vector')
            vect = load_file(v_file)
        else:
            add_to_log('Vectorizing..')
            vect = HashingVectorizer(n_features=2 ** 22, alternate_sign=False, analyzer='word',
                                     decode_error='ignore', token_pattern=r'\b\w{1,}[^\d\W]+\b',
                                     ngram_range=(2, 2))

        if os.path.exists(mtrx):
            X_train = load_file(mtrx)
        else:
            vect_start_time = time.time()

            # For HashVectors
            X_train = vect.transform(text_train)

            vectorization_time = time.time() - vect_start_time
            add_to_log('Vectorization Time: ' + str(vectorization_time))

            v_file = save_vector(vect)
            save_matrix(X_train)

        # Data Synthesis
        X_train, y_train = under_sample_data(X_train, y_train)
        # X_train, y_train = over_sample_data(X_train, y_train)
        # save_sampled_matrix(X_train, y_train)

        # Feature Selection

        selection_start_time = time.time()
        X_short, selectors = feature_selector(X_train, y_train)
        feature_selection_time = time.time() - selection_start_time
        add_to_log('Feature selection time: ' + str(feature_selection_time))
        save_selector(selectors)
        # X_short = X_short.toarray()

        for clf in all_clf:

            summary = 'HV 2,2 2^22 (NV, K-best) Feb2May vs June ' + str(type(clf))
            add_to_log(summary)

            # clf = sync_clf(clf)

            add_to_log('Training Model..')
            training_start_time = time.time()

            # update_clf('2019_Jul_15_09_45')

            c_file = train_classifier(X_short, y_train, clf)
            clf_training_time = time.time() - training_start_time
            add_to_log('Training Time: ' + str(clf_training_time))

            save_selector(selectors)

            add_to_log('Testing Model')
            # test_2019_Jul_02_19_53_mtrx.joblib
            # time_stamp = '2019_Jul_11_17_45'
            tst_mtrx = os.path.join('matrix', 'test_' + time_stamp + '_mtrx.joblib')
            print('test matrix path: ' + tst_mtrx)
            if os.path.exists(tst_mtrx):
                X_test = tst_mtrx

            test_models.test_model(v_file, c_file, test_data_paths, selectors, summary, X_test)


if __name__ == '__main__':
    main()
