from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFE

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


class SurvivalClassifier:
    def __init__(self, train_x, train_y, test_x, verbose=3):
        self.models = []
        self.model_names = []
        self.hyper_parameters = []
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.verbose = verbose

    def append_model(self, model, model_name=None, hyper_parameters=None):
        self.models.append(model)
        self.model_names.append(model_name)
        self.hyper_parameters.append(hyper_parameters)

        return None

    def optimize_models(self, folds=5):
        x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(
            self.train_x, self.train_y, test_size=0.2, random_state=50)

        report_file = open('report{0}.txt'.format(datetime.datetime.now()), 'w+')

        for index, model in enumerate(self.models):
            model_name = self.model_names[index]
            self.log('Model optimization: {0}'.format(model_name), 1)

            optimized_model = GridSearchCV(model, self.hyper_parameters[index], cv=folds,
                                           n_jobs=-1, verbose=self.verbose)

            optimized_model.fit(x_train_cv, y_train_cv)
            optimized_model.predict(x_test_cv)

            best_estimator = optimized_model.best_estimator_

            k_fold_score = np.mean(cross_val_score(
                best_estimator,
                np.append(x_train_cv, x_test_cv, axis=0),
                np.append(y_train_cv, y_test_cv), cv=folds, n_jobs=-1))

            self.log('{0} optimized parameters: {1}'.format(model_name, optimized_model.best_params_), 2)
            self.log('{0} Accuracy: {1}'.format(model_name, optimized_model.score(x_test_cv, y_test_cv)), 1)
            self.log('{0} Accuracy: ({1}-fold): {2}'.format(model_name, str(folds), k_fold_score), 2)

            self.models[index] = best_estimator

            report_file.write('{0} optimized parameters: {1}'.format(model_name, optimized_model.best_params_))

        report_file.close()

        return None

    def plot_learning_curves(self, folds=5):
        for index, model in enumerate(self.models):
            model_name = self.model_names[index]

            plt.figure()
            plt.title(model_name)
            plt.xlabel("Training examples")
            plt.ylabel("Score")

            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=13)

            train_sizes, train_scores, test_scores = learning_curve(
                model, self.train_x, self.train_y, n_jobs=-1, cv=cv)

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plt.grid()
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")

            plt.legend(loc="best")

        plt.show()

        return None

    def accuracy_report(self, folds=5, test_size=0.2):

        x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(
            self.train_x, self.train_y, test_size=test_size, random_state=50)

        for index, model in enumerate(self.models):
            model_name = self.model_names[index]
            self.log('Model report: {0}'.format(model_name), 1)

            k_fold_score = np.mean(cross_val_score(
                model,
                np.append(x_train_cv, x_test_cv, axis=0),
                np.append(y_train_cv, y_test_cv), cv=folds, n_jobs=-1))

            # self.log('{0} Accuracy: {1}'.format(model_name, optimized_model.score(x_test_cv, y_test_cv)), 1)
            self.log('{0} Accuracy: ({1}-fold): {2}'.format(model_name, str(folds), k_fold_score), 2)

        return None

    def feature_selection(self):
        for index, model in enumerate(self.models):
            model_name = self.model_names[index]

            classifier = model
            random_feature_elimination = RFE(classifier)
            random_feature_elimination = random_feature_elimination.fit(self.train_x, self.train_y)

            print('Selected Features for model: {0}'.format(model_name))
            print(self.train_x.ix[:, np.where(random_feature_elimination.support_)[0]].columns.values)

    def get_classifier(self, voting='hard', weights=None):
        model_with_name = []
        for index, model in enumerate(self.models):
            model_with_name.append(tuple([self.model_names[index], model]))

        voting_classifier = VotingClassifier(estimators=model_with_name, voting=voting, weights=weights)
        return voting_classifier

    def learn_predict_flush(self, output_file_name, test_file_name, voting='hard', weights=None):
        voting_classifier = self.get_classifier(voting=voting, weights=weights)
        voting_classifier = voting_classifier.fit(self.train_x, self.train_y)

        ensemble = pd.read_csv(test_file_name)[['PassengerId']].assign(
            Survived=voting_classifier.predict(self.test_x))
        ensemble.to_csv(output_file_name, index=False)

        return None

    def log(self, msg, verbose):
        if self.verbose >= verbose:
            print(msg)

        return None
