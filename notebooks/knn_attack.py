from sklearn.neighbors import KNeighborsClassifier
from experiment_setup import MembershipInference
import numpy as np
import logging


logger = logging.getLogger('xai-privacy')

# define own KNN explainer that provides the k nearest neighbors to a query point
class KnnExplainer():
    def __init__(self, data, outcome_name):
        features = data.drop(outcome_name, axis=1)
        labels = data[outcome_name]
        self._knn_model = KNeighborsClassifier().fit(features, labels)
        self._data = data

    def explain(self, sample_df):
        logger.debug(f'Input sample: {sample_df}')
        nei_indices = self._knn_model.kneighbors(X=sample_df, return_distance=False)
        logger.debug(f'Indices: {nei_indices}')
        neighbors = self._data.iloc[nei_indices[0], :]
        return neighbors


class KnnMembershipInference(MembershipInference):
    def train_explainer(self, data_train, model):
        return KnnExplainer(data_train, self.outcome_name)

    @staticmethod
    def membership_inference_attack_no_model_access(explainer, samples_df):
        inferred_membership = np.empty(len(samples_df))

        for index in range(len(samples_df)):
            # needs double brackets so that iloc returns a dataframe instead of a series
            sample_df = samples_df.iloc[[index], :]

            logger.debug(f'Checking sample {index}: {sample_df.to_numpy()[0]}')

            # explainer does not need target for explanation (remove last column)
            neighbors = explainer.explain(sample_df.drop(sample_df.columns[-1], axis=1))

            logger.debug(f'K nearest neighbors: \n {neighbors.to_numpy()}')

            # check if the sample itself is part of it's own nearest neighbors. In that case, it is part of the training data.
            # otherwise it isn't.
            result = np.isclose(neighbors.to_numpy().astype(float), sample_df.to_numpy().astype(float)).all(
                axis=1).any()

            logger.debug('Inferred membership: %s' % result)
            inferred_membership[index] = result

        return inferred_membership