from experiment_setup import TrainingDataExtraction
import dice_ml
import numpy as np
import random
import pandas as pd
import logging

logger = logging.getLogger('xai-privacy')


class CounterfactualTDE(TrainingDataExtraction):
    def train_explainer(self, data_train, model):
        # train explainer on training data
        d = dice_ml.Data(dataframe=data_train, continuous_features=self.numeric_features, \
                         outcome_name=self.outcome_name)
        m = dice_ml.Model(model=model, backend="sklearn", model_type='classifier')

        # use method "kd-tree" to get counterfactuals drawn from the training data
        return dice_ml.Dice(d, m, method="kdtree")

    @staticmethod
    def training_data_extraction_no_model_access(explainer, num_queries, feature_formats, rng):
        rng = np.random.default_rng(rng)
        seed = rng.integers(100000).item()
        random.seed(seed)

        # Get all feature names
        feature_names = []

        for feature in feature_formats:
            feature_names.append(feature['name'])

        samples_df = pd.DataFrame(columns=feature_names)

        # This is the default number of counterfactuals per query used on the github page of DiCE
        cfs_per_query = 4

        # Generate random samples as queries for the explainer.
        for i in range(num_queries):
            sample = {}
            for feature in feature_formats:
                if feature['isCont']:
                    sample[feature['name']] = rng.integers(feature['min'], feature['max'])
                else:
                    sample[feature['name']] = random.choice(feature['categories'])
            sample_df = pd.DataFrame(sample, index=[0])
            samples_df = pd.concat([samples_df, sample_df], ignore_index=True)

        # Cast categorical features to string again because of DiCE peculiarities
        for feature in feature_formats:
            if not feature['isCont']:
                samples_df[feature['name']] = samples_df[feature['name']].astype(str)
            else:
                samples_df[feature['name']] = samples_df[feature['name']].astype(int)

        # Generate counterfactuals for all random query samples
        e1 = explainer.generate_counterfactuals(samples_df, total_CFs=cfs_per_query, desired_class='opposite')

        # Collect all extracted samples in this dataframe
        extracted_samples_df = pd.DataFrame(columns=feature_names)
        for index in range(len(samples_df)):
            cfs_of_sample = e1.cf_examples_list[index].final_cfs_df
            logger.debug(f'Sample {index}: Counterfactuals \n {cfs_of_sample.to_numpy()}')

            extracted_samples_df = pd.concat([extracted_samples_df, cfs_of_sample], ignore_index=True)

        return extracted_samples_df