from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score

import numpy as np
import pandas as pd


class TitanicSurvivalClassifier:
    def __init__(self, x_train, y_train, models, votes, hyper_parameters, model_names, verbose=0):
        self.x_train = x_train
        self.y_train = y_train
        self.model_ensemble = models
        self.hyper_parameters = hyper_parameters
        self.verbose = verbose
        self.model_names = model_names
        self.votes = votes

    def append_model_to_ensemble(self, model, hyper_parameters, vote, model_name):
        self.model_ensemble.append(model)
        self.hyper_parameters.append(hyper_parameters)
        self.votes.append(vote)
        self.model_names.append(model_name)

        return None

    def optimize_models(self, folds=5, test_set_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(self.x_train,
                                                            self.y_train, test_size=test_set_size,
                                                            random_state=50)

        for index, model in enumerate(self.model_ensemble):
            hyper_parameters = self.hyper_parameters[index]
            model_name = self.model_names[index]

            self.log(1, 'Parameter Optimization for {0}'.format(model_name))

            optimized_model = GridSearchCV(model, param_grid=hyper_parameters, cv=folds, n_jobs=-1)
            optimized_model.fit(x_train, y_train)
            optimized_model.predict(x_test)

            self.log(1, 'Optimized parameters: {0}'.format(optimized_model.best_params_))
            self.log(2, 'Model accuracy (hold-out): {0}'.format(optimized_model.score(x_test, y_test)))

            k_fold_score = np.mean(cross_val_score(
                optimized_model.best_estimator_,
                np.append(x_train, x_test, axis=0),
                np.append(y_train, y_test), cv=folds, n_jobs=-1))

            self.log(2, '{1} accuracy ({0}-fold): {2}'.format(str(folds), model_name, k_fold_score))

            self.model_ensemble[index] = optimized_model.best_estimator_

        return None

    def predict_flush(self, test_data_file_name, x_test, output_file_name):
        model_results = []

        for index, model in enumerate(self.model_ensemble):
            self.log(2, '{0} prediction with params: {1}'.format(self.model_names[index], model))

            model.fit(self.x_train, self.y_train)
            probs = model.predict(np.array(x_test))
            probs[probs == 0] = -1
            model_results.append((probs, self.votes[index]))

        ensemble = pd.read_csv(test_data_file_name)[['PassengerId']].assign(
            Survived=0)

        for probs, votes in model_results:
            for i in range(0, votes):
                ensemble = ensemble.assign(Survived=lambda x: x.Survived + probs)

        (ensemble.assign(Survived=lambda x: np.where(x.Survived > 0, 1, 0))
         .to_csv(output_file_name, index=False))

        return None

    def log(self, dedicated_verbose, message):
        if self.verbose >= dedicated_verbose:
            print(message)

        return None
