import numpy as np
import warnings
import os
import time
from scipy.special import erf

from sklearn.model_selection import RepeatedKFold
from sklearn.utils import check_array, check_X_y, gen_batches
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


class IncrementalWLDA:
    """Incremental Linear Discriminant Analysis (ILDA).

        Incremental Linear Discriminant Analysis is a discriminant model that can be updated as datasets arrive.
        Alternatively it can also be used to improve regular Linear Discriminant Analysis by splitting the inputs
        into batches using the parameter batch_size
    """
    def __init__(self, shrinkage=None, priors=None, n_components=None, batch_size=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.batch_size = batch_size

    def predict(self, X):
        X = check_array(X, dtype=[np.float64, np.float32])
        n_inputs, n_features = X.shape
        # Have to improve
        # check_is_fitted(self)
        if not hasattr(self, 'within_scatter'):
            raise Exception("Model has not been trained yet")

        print('self.within_scatter', self.within_scatter)
        print('self.between_scatter', self.between_scatter)

        lda_matrix = np.dot(np.linalg.pinv(self.within_scatter), self.between_scatter)
        eigVals, eigVecs = np.linalg.eigh(lda_matrix)
        ldaVec = eigVecs[:, np.argsort(eigVals)[::-1]]
        ldaVec /= np.linalg.norm(ldaVec, axis=0)
        # print("X, eigVecs and ldaVec shapes: ", X.shape, eigVecs.shape, ldaVec.shape)
        updatedX = np.dot(X, ldaVec)
        updated_class_means = np.dot(self.class_mean_, ldaVec)

        yVals = np.zeros((n_inputs,))

        # print( "X :::: " , X)
        # print( "Means:::: ", self.means)
        # print( "ldaVec::: ", ldaVec)
        # print("updatedX: ", updatedX.shape)
        # print("updatedMean: ", updated_class_means.shape)

        for i in np.arange(n_inputs):
            currX = np.reshape(updatedX[i, :], (1, n_features))
            # print('updated_class_means.shape', updated_class_means.shape)
            # print('currX.shape', currX.shape)
            distVal = np.linalg.norm(np.subtract(updated_class_means, currX), axis=1)

            # print("distVal: ", distVal)
            # print("amin Value: ", np.amin(distVal, axis=0))
            # print("amin", distVal == np.amin(distVal, axis=0))

            yVals[i] = self.classes_[distVal == distVal.min()]

        print("The given point(s) belong to the following class(es) in the same order: ", yVals)
        # print("Given number of points: ", X.shape[0], " and yVals shape: ", yVals.shape)
        return yVals

    def fit(self, X, y):
        # Test if single fit or multiple fit
        X, y = check_X_y(X, y, estimator=self, ensure_min_samples=1)
        self.classes_ = np.sort(unique_labels(y))
        print('Number of classes: ', self.classes_.size)

        if self.priors is None:  # estimate priors from sample
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            self.priors_ = np.bincount(y_t) / float(len(y))
        else:
            self.priors_ = np.asarray(self.priors)

        if (self.priors_ < 0).any():
            raise ValueError("priors must be non-negative")
        if not np.isclose(self.priors_.sum(), 1.0):
            warnings.warn("The priors do not sum to 1. Renormalizing",
                          UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        # Get the maximum number of components
        if self.n_components is None:
            self._max_components = len(self.classes_) - 1
        else:
            self._max_components = min(len(self.classes_) - 1,
                                       self.n_components)

        # LDA Logic begins here
        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size = 5 * n_features

        for batch in gen_batches(n_samples, self.batch_size):
            self.partial_fit(X[batch], y[batch])

        return self

    def partial_fit(self, X, y, check_input=False):
        n_samples, n_features = X.shape

        print('X.shape', X.shape)
        # This is the first partial_fit
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = 0
            self.class_n_samples_seen_ = np.zeros(self.classes_.shape)

            self.mean_ = np.zeros((1, n_features))
            self.class_mean_ = np.zeros((np.size(self.classes_), n_features))

            self.var_ = .0

            self.between_scatter = np.zeros((n_features, n_features))
            self.within_scatter = np.zeros((n_features, n_features))
            self.class_within_scatter = np.zeros((n_features, n_features, self.classes_.size))

        # If the number of samples is more than 1, we use a batch fit algorithm as in the reference paper Pang et al.
        if n_samples > 1:
            self._batch_fit(X, y, check_input)
        # Else if there is only 1 sample, we use a single fit algorithm as in the reference paper Pang et al.
        else:
            self._single_fit(X, y, check_input)
        return self


    def _single_fit(self, X, y, check_input=False):
        print('Single Fit')
        if check_input:
            X, y = check_X_y(X, y, ensure_min_samples=2, estimator=self)

        # if self.means.size > 0:
        #     totalMean = np.add(np.multiply(self.means, self.numbers), X) / (np.sum(self.numbers) + 1)
        # else:
        #     totalMean = X
        # # New data belongs to Existing Class
        # if y in self.classes:
        #     # Update Mean and Number of datasets in the class
        #     indicesVal, = np.where(self.classes == Y)[0]
        #     yClassMean = self.means[indicesVal, :]
        #     yNums = self.numbers[indicesVal]
        #     yClassMean = np.add((np.asarray(yClassMean) * yNums), X) / (yNums + 1)
        #     yClassMean = np.asarray(yClassMean)[0,:]
        #     self.means[indicesVal,:] = yClassMean
        #     self.numbers[indicesVal] = yNums + 1
        #
        #     # Update S_b
        #     updatedS_b = np.zeros((X.shape[1], X.shape[1]))
        #     idx = 0
        #     for classVal in self.classes:
        #         if classVal == y:
        #             thisClassDiff = np.subtract(yClassMean, totalMean)
        #         else:
        #             thisClassDiff = np.subtract(self.means[idx, :], totalMean)
        #         updatedS_b += self.numbers[idx] * np.dot(thisClassDiff, thisClassDiff.T)
        #         idx += 1
        #     self.s_b = updatedS_b
        #
        #     # Update S_w
        #     interClassDiff = np.subtract(X, yClassMean)
        #     if (self.s_w.size > 0):
        #         self.s_w = np.add(self.s_w, ((yNums / (yNums + 1)) *
        #                                      np.dot(interClassDiff.T,
        #                                                  interClassDiff)))
        #     else:
        #         self.s_w = (yNums / (yNums + 1)) * \
        #                    np.dot(interClassDiff.T, interClassDiff)
        # # New data belongs to New Class
        # else:
        #     # Update formulae
        #     if self.means.size > 0:
        #         self.means = np.vstack((self.means, X))
        #         self.numbers = np.append(self.numbers, 1)
        #         self.s_b = self.s_b + np.dot(X - totalMean, (X - totalMean).T)
        #         self.classes = np.append(self.classes, unique_labels(y))
        #     # Create formulae
        #     else:
        #         self.means = X
        #         self.numbers = np.array([1])
        #         self.s_b = np.dot(X - totalMean, (X - totalMean).T)
        #         self.classes = unique_labels(y)

    def _batch_fit(self, X, y, check_input=False):
        print('Batch fit')
        if check_input:
            X, y = check_X_y(X, y, ensure_min_samples=2, estimator=self)

        current_n_samples, n_features = X.shape
        # Update stats - they are 0 if this is the first step
        updated_mean, updated_var, updated_n_samples_seen_ = _incremental_mean_and_var(X, last_mean=self.mean_,
                                                                           last_variance=self.var_,
                                                                           last_sample_count=self.n_samples_seen_)
        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X = np.subtract(X, updated_mean)
        else:
            col_batch_mean = np.mean(X, axis=0)
            X = np.subtract(X, col_batch_mean)

        # Updating algorithm
        # Updating Mean and Class Means
        updated_class_mean = self.class_mean_
        updated_class_n_samples_seen_ = self.class_n_samples_seen_
        # print('updated_class_n_samples_seen_', updated_class_n_samples_seen_)
        # print('updated_class_mean', updated_class_mean)
        for i, current_class in enumerate(self.classes_):
            current_class_samples = X[y == current_class, :]
            n_current_class_samples = current_class_samples.shape[0]
            previous_n_class_samples = updated_class_n_samples_seen_[i]
            if n_current_class_samples > 0 and previous_n_class_samples > 0:
                previous_class_sum_current_class = updated_class_mean[i, :] * updated_class_n_samples_seen_[i]
                current_class_sum_current_class = np.sum(current_class_samples, axis=0)

                # print('previous_class_sum_current_class.shape', previous_class_sum_current_class.shape)
                # print('current_class_sum_current_class.shape', current_class_sum_current_class.shape)
                # print('updated_class_mean.shape', updated_class_mean.shape)
                # print('updated_class_n_samples_seen_.shape', updated_class_n_samples_seen_[i])

                updated_class_n_samples_seen_[i] += n_current_class_samples
                updated_class_mean[i, :] = (previous_class_sum_current_class + current_class_sum_current_class) / \
                                           updated_class_n_samples_seen_[i]
            elif n_current_class_samples > 0:
                updated_class_mean[i, :] = np.mean(current_class_samples, axis=0)
                updated_class_n_samples_seen_[i] = n_current_class_samples

        updated_class_within_scatter = self.class_within_scatter
        for i, current_class_mean in enumerate(updated_class_mean):
            current_class_samples = X[y == self.classes_[i], :]
            n_current_class_samples = current_class_samples.shape[0]
            l_c = current_class_samples.shape[0]
            n_c = self.class_n_samples_seen_[i]
            mean_y_c = np.reshape(np.mean(current_class_samples, axis=0), (n_features, 1))

            if n_current_class_samples > 0 and n_c > 0:
                # print('current_class_samples.shape', current_class_samples.shape)
                mean_x_c = np.reshape(self.class_mean_[i, :], (n_features, 1))

                D_c = (mean_y_c - mean_x_c).dot((mean_y_c - mean_x_c).T)

                E_c = np.zeros(D_c.shape)
                for current_samples, j in enumerate(current_class_samples):
                    E_c += (current_samples - mean_x_c).dot((current_samples - mean_x_c).T)

                F_c = np.zeros(D_c.shape)
                for current_samples, j in enumerate(current_class_samples):
                    F_c += (current_samples - mean_y_c).dot((current_samples - mean_y_c).T)

                updated_class_within_scatter[:, :, i] += ((n_c * l_c * l_c) * D_c / np.square(n_c + l_c)) + \
                                                         ((np.square(n_c) * E_c) / np.square(n_c + l_c)) + \
                                                         ((l_c * (l_c + (2 * n_c)) * F_c) / np.square(n_c + l_c))
            elif n_current_class_samples > 0:
                updated_class_within_scatter[:, :, i] = (current_class_samples - mean_y_c).dot(
                    (current_class_samples - mean_y_c).T)
        updated_within_scatter = np.sum(updated_class_within_scatter, axis=2)

        # Updating between class scatter
        updated_between_scatter = self.between_scatter
        for i, i_class_mean in enumerate(updated_class_mean[:-1, :]):
            for j_class_mean in updated_class_mean[i+1:, :]:
                print('Computing mean difference of means:::', i_class_mean, j_class_mean)
                current_mean_difference = np.reshape(i_class_mean - j_class_mean, (1, n_features))
                d_ij = current_mean_difference.dot(np.linalg.pinv(updated_within_scatter)).dot(current_mean_difference.T)
                d_ij = np.sqrt(d_ij)
                if d_ij > 0:
                    w_d_ij = erf(d_ij / (2 * np.sqrt(2))) / (2 * np.square(d_ij))
                    # print('current_mean_difference.shape', current_mean_difference.shape)
                    updated_between_scatter += w_d_ij * current_mean_difference.T.dot(current_mean_difference)
            # n = X[y == self.classes_[i], :].shape[0]
            # current_class_mean = current_class_mean.reshape(1, n_features)
            # updated_mean = updated_mean.reshape(1, n_features)
            # if n > 0:
            #     updated_between_scatter += n * (current_class_mean - updated_mean).T.dot(
            #         current_class_mean - updated_mean)

        # if np.any(np.isnan(updated_between_scatter)):
        #     print('Reached nan:::: ', n)
        #     print('Updatec class mean:::', updated_class_mean)
        #     print('updated mean::::', updated_mean)

        # Final values after computation
        self.n_samples_seen_ = updated_n_samples_seen_
        self.class_n_samples_seen_ = updated_class_n_samples_seen_
        self.mean_ = updated_mean
        self.class_mean_ = updated_class_mean
        self.var_ = updated_var
        self.between_scatter = updated_between_scatter
        self.within_scatter = updated_within_scatter
        self.class_within_scatter = updated_class_within_scatter


def run_on_iris():
    # Load the dataset
    data = datasets.load_iris()
    #
    X = data.data
    y = data.target
    start_time = time.time()
    path = "Datasets/AR/"
    # X = np.load(os.path.join(path, 'X.npy'))
    # y = np.load(os.path.join(path, 'Y.npy'))

    n_splits = 5
    n_repeats = 1
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    accuracy_values = []

    count_val = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit and predict using LDA
        ilda = IncrementalWLDA(batch_size=25)
        print("Fitting")
        ilda.fit(X_train, y_train)

        print("Predicting")
        y_train_predicted = ilda.predict(X_train)
        y_test_predicted = ilda.predict(X_test)

        accuracy = accuracy_score(y_train, y_train_predicted)
        print("Train Accuracy:", accuracy)

        accuracy = accuracy_score(y_test, y_test_predicted)
        print("Test Accuracy:", accuracy)

        accuracy_values.append(accuracy)
        count_val = count_val + 1

    print('Accuracy Values:')
    print(accuracy_values)

    average_accuracy = np.average(accuracy_values)
    print('Average accuracy:')
    print(average_accuracy)

    f = open(os.path.join(path, 'average_accuracy_iris.txt'), 'w')
    f.write('Average Accuracy on IRIS data= ' + str(average_accuracy) + '\n')
    f.close()
    print("--- Total time taken %s seconds ---" % (time.time() - start_time))
    # pca = PCA()
    # pca.plot_in_2d(X_test, y_pred, title="LDA", accuracy=accuracy)


def run_on_data(path):
    start_time = time.time()
    X = np.load(os.path.join(path, 'X.npy'))
    y = np.load(os.path.join(path, 'Y.npy'))

    n_splits = 5
    n_repeats = 1
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    accuracy_values = []

    if X.shape[1] > 10000:
        pca = PCA()
        X = pca.fit_transform(X)

    count_val = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit and predict using LDA
        ilda = IncrementalWLDA(batch_size=50)
        print("Fitting")
        ilda.fit(X_train, y_train)

        print("Predicting")
        y_train_predicted = ilda.predict(X_train)
        y_test_predicted = ilda.predict(X_test)

        accuracy = accuracy_score(y_train, y_train_predicted)
        print("Train Accuracy:", accuracy)

        accuracy = accuracy_score(y_test, y_test_predicted)
        print("Test Accuracy:", accuracy)

        accuracy_values.append(accuracy)
        count_val = count_val + 1

    print('Accuracy Values:')
    print(accuracy_values)

    average_accuracy = np.average(accuracy_values)
    print('Average accuracy:')
    print(average_accuracy)

    f = open(os.path.join(path, 'ilda_accuracy.txt'), 'w')
    f.write('Average Accuracy on AR data = ' + str(average_accuracy) + '\n')
    f.close()
    print("--- Total time taken %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    ar_path = "Datasets/AR/"
    # cacd_path = "Datasets/CACD/"
    # run_on_iris()
    run_on_data(ar_path)
    # run_on_data(cacd_path)
    # ar_data_test_pca()
    # iris_data()
