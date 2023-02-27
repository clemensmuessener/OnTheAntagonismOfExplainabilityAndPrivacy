from experiment_setup import MembershipInference
from sklearn.inspection import partial_dependence
import numpy as np
import logging

logger = logging.getLogger('xai-privacy')


class PDPMembershipInference(MembershipInference):
    def train_explainer(self, data_train, model):
        # Generate partial dependence plots for each feature (only for training data)
        pd_plots = []

        for i in range(self.features.shape[1]):
            # grid resolution must be high enough so that each unique feature value in the training data gets its own
            # partial dependence value on the y axis (per our definition of PDP)
            pd_plots.append(partial_dependence(estimator=model, X=data_train.drop(self.outcome_name, axis=1),
                                               features=[i], percentiles=(0, 1), grid_resolution=len(data_train) + 1,
                                               kind='average'))

        return pd_plots

    @staticmethod
    def membership_inference_attack_no_model_access(explainer, samples_df):
        pd_plots = explainer
        # we only use the features for membership inference, not the target
        samples = samples_df.to_numpy()[:, :-1]
        num_features = samples.shape[1]

        results = np.empty(samples.shape[0])

        for i, sample in enumerate(samples):
            logger.debug(f'Checking sample {i}: {sample}')

            is_member = True

            for j in range(num_features):
                # check if j-th feature value of sample is contained in j-th partial dependence plot:
                feature_values_in_plot = pd_plots[j]['values'][0]
                feature_value_in_sample = sample[j]

                if feature_value_in_sample in feature_values_in_plot:
                    logger.debug(f'Feature {j} ({feature_value_in_sample}) is contained in respective PDP.')
                    continue
                else:
                    logger.debug(f'Feature {j} ({feature_value_in_sample}) is NOT contained in respective PDP.')
                    is_member = False
                    break

            logger.debug(f'Inferred membership as {is_member}')
            results[i] = is_member

        return results