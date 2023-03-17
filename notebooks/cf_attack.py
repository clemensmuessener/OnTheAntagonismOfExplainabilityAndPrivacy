from experiment_setup import MembershipInference, TrainingDataExtraction
import dice_ml
import numpy as np
import random
import pandas as pd
import logging

logger = logging.getLogger('xai-privacy')


class CounterfactualMembershipInference(MembershipInference):
    def train_explainer(self, data_train, model):
        # train explainer on training data
        d = dice_ml.Data(dataframe=data_train, continuous_features=self.numeric_features, \
                         outcome_name=self.outcome_name)
        m = dice_ml.Model(model=model, backend="sklearn", model_type='classifier')

        # use method "kd-tree" to get counterfactuals drawn from the training data
        return dice_ml.Dice(d, m, method="kdtree")

    @staticmethod
    def membership_inference_attack_no_model_access(explainer, samples_df):
        num_samples = len(samples_df)
        inferred_membership = np.full(num_samples, False)
        # we only use the features for membership inference, not the target. Therefore we must drop the last column.
        samples_df = samples_df.drop(samples_df.columns[-1], axis=1)

        # This is the default number of counterfactuals per query used on the github page of DiCE
        cfs_per_query = 4

        # get first counterfactuals for all given samples
        e1 = explainer.generate_counterfactuals(samples_df, total_CFs=cfs_per_query, desired_class='opposite')

        # collect all first counterfactuals in this dataframe to plug it into the explainer once more
        first_cfs_all = pd.DataFrame(columns=samples_df.columns)

        # collect the original sample index corresponding to a first counterfactual
        # this is necessary in order to remember which first counterfactuals belonged to which original sample
        respective_sample_index = []

        # collect first counterfactuals
        for index in range(num_samples):
            # get counterfactuals for given sample:
            first_cfs = e1.cf_examples_list[index].final_cfs_df
            logger.debug(f'Sample {index}: 1st counterfactuals: \n {first_cfs.to_numpy()}')

            first_cfs_all = pd.concat([first_cfs_all, first_cfs])

            for i in range(len(first_cfs)):
                respective_sample_index.append(index)

        respective_sample_index = np.array(respective_sample_index)

        # get second counterfactuals for all first counterfactuals
        e2 = explainer.generate_counterfactuals(first_cfs_all, total_CFs=cfs_per_query, desired_class='opposite')

        # compare all second counterfactuals with the samples they were generated for
        for i, second_cfs_obj in enumerate(e2.cf_examples_list):
            # get the sample that these second counterfactuals belong to:
            index = respective_sample_index[i]
            sample_df = samples_df.iloc[[index], :]

            logger.debug(f'Sample {index}: {sample_df.to_numpy()}')

            second_cfs = second_cfs_obj.final_cfs_df

            logger.debug(f'Sample {index}: 2nd counterfactuals: \n {second_cfs.to_numpy()}')

            # if any counter-counterfactual is equal to the given sample, then it is part of the training data:
            # np.isclose is used for comparison because explainer may round floating point values
            result = np.isclose(second_cfs.to_numpy().astype(float), sample_df.to_numpy().astype(float)).all(
                axis=1).any()

            if result:
                logger.debug(f'Inferred membership as true.')
                inferred_membership[index] = True

        return inferred_membership


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