import logging
from experiment_setup import MembershipInference
from sklearn.inspection import partial_dependence
import numpy as np
import pandas as pd

logger = logging.getLogger('xai-privacy')


class IceMembershipInference(MembershipInference):
    def train_explainer(self, data_train, model):
        # Calculate ICE
        ice_features = []

        for i in range(self.features.shape[1]):
            logger.debug(f'Calculating ICE for feature {i}')
            ice_features.append(partial_dependence(estimator=model, X=data_train.drop(self.outcome_name, axis=1),
                                                   features=[i], percentiles=(0, 1), kind='individual'))

        return ice_features

    @staticmethod
    def membership_inference_attack_model_access(explainer, samples_df, model):
        # we only use the features for membership inference, not the target. Therefore we must drop the last column.
        samples_df = samples_df.drop(samples_df.columns[-1], axis=1)
        samples = samples_df.to_numpy()
        ice_features = explainer
        pred_func = model.predict_proba
        columns = samples_df.columns

        results = np.empty(samples.shape[0])

        for i in range(samples.shape[0]):

            sample = samples[i]

            logger.debug(f'Checking sample {i}: {sample}')

            for num, ice in enumerate(ice_features):

                # Get the grid for this feature. These are the points at which we will get predictions from the model.
                feature_values = ice['values'][0]

                # Create array that contains the sample as many times as there are different grid points for this feature.
                changed_samples = np.tile(sample, (feature_values.shape[0], 1))

                # Change each copied samples feature in question to a point from the grid.
                changed_samples[:, num] = feature_values

                # create pandas dataframe from numpy array
                changed_samples_df = pd.DataFrame(changed_samples, columns=columns)

                # Get the predictions for these changed samples.
                preds = pred_func(changed_samples_df)[:, 1]

                # First, create array with boolean values whether the recreated predictions and the actual ICE predictions match.
                # Then check if there is any row where all elements are true.
                # That means there is at least one ICE line with identical predictions.
                res = np.isclose(preds, ice['individual'][0], rtol=1e-03).all(1).any()

                # If there is no identical ICE line for this feature, the sample can't be part of the training data.
                # Break out of loop. res is false and will be appended to the list of results.
                if res == False:
                    logger.debug(f'Recreated ICE line for feature {num} does not match.')
                    break
                logger.debug(f'Recreated ICE line for feature {num} matches.')

            logger.debug(f'Inferred membership as {res}')

            # This appends the last res to the results list. If a matching line for every feature was found, this is true
            # (sample is in training data). Otherwise the last res will be false (sample is not in training data)
            results[i] = res

        return results