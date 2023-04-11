from experiment_setup import MembershipInference
import logging
import shap
from matplotlib import pyplot as plt
import numpy as np

logger = logging.getLogger('xai-privacy')


# This class is necessary so that we don't need to pass a function to the explainer. That would create issues when pickling for parallelization.
class ShapModel():
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        return self.model.predict(X)


class ShapMembershipInference(MembershipInference):
    def train_explainer(self, data_train, model):
        # Generate SHAP dependence plots for each feature (only for training data)
        logger.debug(f'Calculating SHAP values for dependence plots...')

        x_train = data_train.drop(self.outcome_name, axis=1)
        num_of_features = x_train.shape[1]

        shap_model = ShapModel(model)

        explainer = shap.explainers.Exact(shap_model, x_train)

        shap_values = explainer(x_train)

        d_plots_for_features = []

        for i in range(num_of_features):
            logger.debug(f'Calculating dependence plot for feature {i} of {num_of_features}')
            fig, ax = plt.subplots()

            # calculate SHAP dependence plot for the i-th feature using the shap values for the heart disease likelihood output
            # do not show the plot, do not include interactions
            shap.dependence_plot(i, shap_values.values, x_train, show=False, interaction_index=None, ax=ax)

            # x, y data of plot
            path_collection_offsets = ax.collections[0].get_offsets()
            d_plots_for_features.append(path_collection_offsets)

            # close current figure
            plt.close()

        return {'d_plots': d_plots_for_features, 'shap_value_explainer': explainer}

    @staticmethod
    def membership_inference_attack_no_model_access(explainer, samples_df):
        d_plots_for_features = explainer['d_plots']
        shap_value_explainer = explainer['shap_value_explainer']
        samples_df = samples_df.astype(float)
        # we only use the features for membership inference, not the target. Therefore we must drop the last column.
        samples_df = samples_df.drop(samples_df.columns[-1], axis=1)

        inferred_membership = np.empty(len(samples_df))

        for index in range(len(samples_df)):
            # needs double brackets so that iloc returns a dataframe instead of a series
            sample = samples_df.iloc[[index], :]

            logger.debug(f'Checking sample {index}: {sample.to_numpy()[0]}')

            shap_values = shap_value_explainer(sample)

            is_member = True

            for i, dependence_plot in enumerate(d_plots_for_features):

                # check if datapoint (i-th feature value, i-th SHAP value) of sample is contained in i-th shap dependence plot:
                feature_value = sample.iloc[0, i]
                shap_value = shap_values.values[0, i]

                close_rows = np.isclose(dependence_plot, np.array([feature_value, shap_value])).all(axis=1)

                if close_rows.any():
                    logger.debug(
                        f'Datapoint for feature {i} ({feature_value}, {shap_value}) is in respective plot at indices {np.where(close_rows)[0]}.')
                    continue
                else:
                    logger.debug(
                        f'Datapoint for feature {i} ({feature_value}, {shap_value}) is NOT in respective plot.')
                    is_member = False
                    break

            logger.debug(f'Inferred membership as {is_member}')
            inferred_membership[index] = is_member

        return inferred_membership