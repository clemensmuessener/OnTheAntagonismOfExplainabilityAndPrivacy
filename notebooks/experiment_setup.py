import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging
import random
import multiprocessing
import time
import datetime

logger = logging.getLogger('xai-privacy')


def get_heart_disease_dataset(halve_dataset=False):

    logger.info('Loading dataset 1: heart disease (numeric features) ...')

    columns_heart = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR',
                     'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'CA', 'Thal', 'HeartDisease']

    filename_cleveland = '../data/processed.cleveland.data'
    data_cleveland = pd.read_csv(filename_cleveland, names=columns_heart)

    filename_hungarian = '../data/processed.hungarian.data'
    data_hungarian = pd.read_csv(filename_hungarian, names=columns_heart)

    filename_switzerland = '../data/processed.switzerland.data'
    data_switzerland = pd.read_csv(filename_switzerland, names=columns_heart)

    filename_va = '../data/processed.va.data'
    data_va = pd.read_csv(filename_va, names=columns_heart)

    filename_stalog = '../data/heart.dat'
    data_stalog = pd.read_csv(filename_stalog, sep=' ', names=columns_heart)

    numeric_features_heart = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    all_features_heart = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR',
                          'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'CA', 'Thal']
    outcome_name_heart = 'HeartDisease'


    def fix_target(df):
        df.loc[(df[outcome_name_heart] > 0), outcome_name_heart] = 1
        return df


    data_cleveland = fix_target(data_cleveland)
    data_switzerland = fix_target(data_switzerland)
    data_va = fix_target(data_va)

    data_stalog.loc[(data_stalog[outcome_name_heart] == 1), outcome_name_heart] = 0
    data_stalog.loc[(data_stalog[outcome_name_heart] == 2), outcome_name_heart] = 1

    data_heart = pd.concat([data_cleveland, data_hungarian, data_switzerland, data_va, data_stalog])
    data_heart = data_heart.sample(frac=1, random_state=42).reset_index(drop=True)

    for feature in all_features_heart:
        if feature in numeric_features_heart:
            # remove rows with missing numeric value
            non_empty_indices = data_heart[feature] != '?'
            len_before_removal = len(data_heart)
            data_heart = data_heart[non_empty_indices]
            print(f'Feature {feature}: removed {len_before_removal - len(data_heart)} rows for missing values.')
        else:
            # add category "unknown" if categorical feature with missing values
            empty_indices = data_heart[feature] == '?'
            if empty_indices.any():
                unique_values = data_heart[feature].unique().tolist()
                unique_values.remove('?')
                unique_values = [float(i) for i in unique_values]
                max_category = max(unique_values)
                unknown_category = max_category + 1
                data_heart[feature] = data_heart[feature].replace('?', unknown_category)
                print(f'Feature {feature}: add unknown category {unknown_category}')

    data_heart = data_heart.astype(float)

    if halve_dataset:
        data_heart = data_heart.sample(frac=0.5, random_state=42).reset_index(drop=True)

    data_heart_num = data_heart.drop('Sex', axis=1).drop('ChestPainType', axis=1).drop('FastingBS', axis=1).drop(
        'RestingECG', axis=1).drop('ExerciseAngina', axis=1).drop('ST_Slope', axis=1).drop('CA', axis=1).drop('Thal',
                                                                                                              axis=1)

    numeric_features_heart_num = numeric_features_heart
    all_features_heart_num = numeric_features_heart_num

    data_heart_cat = data_heart.copy()
    for feature in numeric_features_heart:
        # we discretize the numeric features into 10 bins of equal width
        data_heart_cat[feature] = pd.cut(data_heart_cat[feature], 10)
        # represent categories as numbers (expected by experiment code later on)
        data_heart_cat[feature] = OrdinalEncoder(dtype=float).fit_transform(data_heart_cat[[feature]])

    numeric_features_heart_cat = []
    all_features_heart_cat = all_features_heart

    data_heart_cat.head(5)

    len_heart_before = len(data_heart)
    len_heart_num_before = len(data_heart_num)
    len_heart_cat_before = len(data_heart_cat)

    data_heart = data_heart.drop_duplicates(subset=all_features_heart)
    data_heart_num = data_heart_num.drop_duplicates(subset=all_features_heart_num)
    data_heart_cat = data_heart_cat.drop_duplicates(subset=all_features_heart_cat)

    print(f'Dropped {len_heart_before - len(data_heart)} of {len_heart_before}')
    print(f'Dropped {len_heart_num_before - len(data_heart_num)} of {len_heart_num_before}')
    print(f'Dropped {len_heart_cat_before - len(data_heart_cat)} of {len_heart_cat_before}')

    data_heart_dict = {'name': 'heart', 'dataset': data_heart, 'num': numeric_features_heart, 'outcome': outcome_name_heart}
    data_heart_num_dict = {'name': 'heart numeric', 'dataset': data_heart_num, 'num': numeric_features_heart_num,
                           'outcome': outcome_name_heart}
    data_heart_cat_dict = {'name': 'heart categorical', 'dataset': data_heart_cat, 'num': numeric_features_heart_cat,
                           'outcome': outcome_name_heart}

    return data_heart_dict, data_heart_num_dict, data_heart_cat_dict


def get_census_dataset(halve_dataset=False):
    logger.info('Loading dataset 2: census income (categorical features) ...')

    filename_census = '../data/adult.data.csv'

    all_features_census = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', \
                           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                           'native_country', 'label']

    data_census = pd.read_csv(filename_census, names=all_features_census)

    len_before = len(data_census)

    data_census = data_census[data_census.workclass != ' ?']
    data_census = data_census[data_census.native_country != ' ?']
    data_census = data_census[data_census.occupation != ' ?']

    print(f'Dropped: {len_before - len(data_census)} of {len_before}')

    data_census[
        ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']] = \
        OrdinalEncoder(dtype=int).fit_transform(
            data_census[['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', \
                         'native_country']])

    data_census['income'] = LabelEncoder().fit_transform(data_census['label'])

    data_census = data_census.drop('label', axis=1)
    data_census = data_census.drop('fnlwgt', axis=1)

    numeric_features_census = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

    all_features_census = ['age', 'workclass', 'education', 'education_num', 'marital_status', 'occupation', \
                           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                           'native_country']

    outcome_name_census = 'income'

    if halve_dataset:
        data_census = data_census.sample(frac=0.5, random_state=42).reset_index(drop=True)

    data_census_num = data_census.drop('workclass', axis=1).drop('education', axis=1).drop('marital_status', axis=1).drop(
        'occupation', axis=1).drop('relationship', axis=1).drop('race', axis=1).drop('sex', axis=1).drop('native_country',
                                                                                                         axis=1)

    numeric_features_census_num = numeric_features_census
    all_features_census_num = numeric_features_census_num

    data_census_cat = data_census.copy()
    for feature in numeric_features_census:
        # we discretize the numeric features into 10 bins of equal width
        data_census_cat[feature] = pd.cut(data_census_cat[feature], 10)
        # represent categories as numbers (expected by experiment code later on)
        data_census_cat[feature] = OrdinalEncoder(dtype=int).fit_transform(data_census_cat[[feature]])

    numeric_features_census_cat = []
    all_features_census_cat = all_features_census

    len_census_before = len(data_census)
    len_census_num_before = len(data_census_num)
    len_census_cat_before = len(data_census_cat)

    data_census = data_census.drop_duplicates(subset=all_features_census)
    data_census_num = data_census_num.drop_duplicates(subset=all_features_census_num)
    data_census_cat = data_census_cat.drop_duplicates(subset=all_features_census_cat)

    print(f'census: Dropped {len_census_before - len(data_census)} of {len_census_before}')
    print(f'num: Dropped {len_census_num_before - len(data_census_num)} of {len_census_num_before}')
    print(f'cat: Dropped {len_census_cat_before - len(data_census_cat)} of {len_census_cat_before}')

    data_census_dict = {'name': 'census', 'dataset': data_census, 'num': numeric_features_census,
                        'outcome': outcome_name_census}
    data_census_num_dict = {'name': 'census numeric', 'dataset': data_census_num, 'num': numeric_features_census_num,
                            'outcome': outcome_name_census}
    data_census_cat_dict = {'name': 'census categorical', 'dataset': data_census_cat, 'num': numeric_features_census_cat,
                            'outcome': outcome_name_census}

    return data_census_dict, data_census_num_dict, data_census_cat_dict


class XaiPrivacyExperiment():
    """Generic framework for an XAI and data privacy experiment

    Attributes
    ----------
    rs, rng
        Random states for numpy
    data
        Pandas dataframe of the dataset that the experiment is executed on. Contains features and labels.
    numeric_features : list[str]
        The numeric feature names of the dataset.
    categorical_features : list[str]
        The categorical feature names of the dataset.
    outcome_name : str
        The name of the column that contains the labels.
    features
        Pandas dataframe that only contains the feature values of all samples (not labels).
    labels
        Pandas dataframe that only contains the labels of all samples (not features).

    Methods
    -------
    train_explainer(data_train, model):
        Trains the explainer on the given data and model (abstract method).

    """

    def __init__(self, data, numeric_features, outcome_name, random_state: int):
        """
        Parameters
        ----------
        data
            Pandas dataframe of the dataset that the experiment is executed on. Contains features and labels.
        numeric_features : list[str]
            The numeric feature names of the dataset.
        outcome_name : str
            The name of the column that contains the labels.
        random_state: int
            The seed for all random actions during the experiment (such as drawing samples for membership inference)
        """
        # create random state from seed. This will be used for all random actions (such as drawing samples for membership inference)
        self.rs = np.random.RandomState(seed=random_state)
        self.rng = np.random.default_rng(random_state)
        random.seed(random_state)

        self.data = data.sample(frac=1, random_state=self.rng)
        self.numeric_features = numeric_features
        self.outcome_name = outcome_name

        # split dataset into features and labels.
        self.features = self.data.drop(outcome_name, axis=1)
        self.labels = self.data[outcome_name]

        # names of the categorical features
        self.categorical_features = self.features.columns.difference(numeric_features).tolist()

        logger.debug(f'Numeric Features: {self.numeric_features}')
        logger.debug(f'Categorical Features: {self.categorical_features}')

    def _model_pipeline(self, model):
        if len(self.categorical_features) > 0 and len(self.numeric_features) > 0:
            # Define transformer to one-hot-encode categorical features and numeric features are scaled
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, self.numeric_features),
                    ("cat", categorical_transformer, self.categorical_features),
                ]
            )

        elif len(self.categorical_features) > 0:
            # Define transformer to one-hot-encode categorical features
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", categorical_transformer, self.categorical_features)
                ]
            )

        else:
            # Define transformer to scale numeric features
            numeric_transformer = StandardScaler()

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, self.numeric_features)
                ]
            )

        return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    def _print_times(self, start_time, model_time, explainer_time, end_time):
        print(
            f'Total time: {end_time - start_time:.2f}s (training model: {model_time - start_time:.2f}s, training explainer: {explainer_time - model_time:.2f}s, experiment: {end_time - explainer_time:.2f}s)')

    def train_explainer(self, data_train, model):
        """Trains the explainer on the given data and model

        Abstract method that must be implemented by subclass. Returns the explainer.

        Parameters
        ----------
        data_train
            The training data (features and labels).
        model
            The trained model that will be explained by the explainer.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclass.
        """

        raise NotImplementedError


class MembershipInference(XaiPrivacyExperiment):
    """
    Executes a membership inference attack. Some public methods must be implemented by subclass.

    Methods
    -------
    membership_inference_experiment(num_queries: int, model, model_access: bool)
        Executes membership inference experiment
    membership_inference_attack_model_access(explainer, samples_df, model):
        Executes membership inference attack with access to the model
    membership_inference_attack_no_model_access(explainer, samples_df):
        Executes membership inference attack without access to the model
    """

    def membership_inference_experiment(self, num_queries: int, model, model_access: bool, threads,
                                        pretrained_model_and_explainer=None):
        """Executes membership inference experiment

        Executes the membership inference experiment with the dataset that this object was instantiated with. Trains given
        model on half the dataset and tests accuracy, precision and recall of the implemented membership inference attack.
        If model_access is True, the attack method with the parameter "model" is used (the attacker has access to the model).
        Otherwise, the attack method without that parameter is used (the attacker has no access to the model).

        Parameters
        ----------
        num_queries : int
            Number of samples that the membership inference attack is attempted on. Should not be greater than len(data).
            If None, then membership inference will be attemped on all samples.
        model
            The untrained model used in the experiment
        model_access : bool
            Whether the membership inference attack is executed with attacker access to the model or without.
        """
        # stop the time of training model, training explainer, and executing experiment
        start_time = time.time()

        if pretrained_model_and_explainer is None:
            model, explainer, train_model_time, train_explainer_time, data_train, data_test = self._train_model_and_explainer(
                model)
        else:
            model, explainer, train_model_time, train_explainer_time = pretrained_model_and_explainer
            data_train, data_test = self._split_data()

        # draw samples from training and test data. record each sample's membership in training data.
        samples_df, actual_membership = self._draw_mi_samples(num_queries, data_train, data_test)

        # infer membership using membership inference attack against the explainer
        if threads > 1:
            arg_list = self._args_for_parallel_execution(threads, samples_df, explainer, model, model_access)

            if model_access:
                with multiprocessing.Pool(threads) as p:
                    results = p.starmap(self.membership_inference_attack_model_access, arg_list)
            else:
                with multiprocessing.Pool(threads) as p:
                    results = p.starmap(self.membership_inference_attack_no_model_access, arg_list)

            inferred_membership = np.concatenate(results, axis=0)
        else:
            if model_access:
                inferred_membership = self.membership_inference_attack_model_access(explainer, samples_df, model)
            else:
                inferred_membership = self.membership_inference_attack_no_model_access(explainer, samples_df)

        end_time = time.time()
        self._print_times(start_time, train_model_time, train_explainer_time, end_time)

        # calculate accuracy, precision and recall
        return self._calc_accuracy_precision_recall(actual_membership, inferred_membership)

    def _train_model_and_explainer(self, model):
        # create pipeline that transforms categorical features to one hot encoding
        model = self._model_pipeline(model)

        # split data into two halves (one is used for training and inference, the other only for inference)
        data_train, data_test = self._split_data()

        # train classifier on training data
        model = model.fit(data_train.drop(self.outcome_name, axis=1), data_train[self.outcome_name])
        train_model_time = time.time()

        # train explainer on training data and classifier
        explainer = self.train_explainer(data_train, model)
        train_explainer_time = time.time()

        return model, explainer, train_model_time, train_explainer_time, data_train, data_test

    def _split_data(self):
        # split data into two halves. One is used for training, the other as test data that is not part of the training data.
        # this test data will be needed as membership inference samples that do not belong to the training data.
        idx_mid = int(self.features.shape[0] / 2)

        data_train = self.data.iloc[idx_mid:, :]
        data_test = self.data.iloc[:idx_mid, :]

        # remove test samples that have a category that is not covered by the training samples
        len_test_prev = len(data_test)
        for feature in self.features.columns:
            if feature in self.categorical_features:
                unique_train = data_train[feature].unique().tolist()
                unique_test = data_test[feature].unique().tolist()

                values_not_in_train = [x for x in unique_test if x not in unique_train]
                for value in values_not_in_train:
                    data_test = data_test[data_test[feature] != value]
            else:
                min_train = data_train[feature].min()
                max_train = data_train[feature].max()

                data_test = data_test[(data_test[feature] >= min_train) & (data_test[feature] <= max_train)]

        logger.debug(f'Removed {len_test_prev - len(data_test)} test samples due to unknown category.')

        return data_train, data_test

    def _draw_mi_samples(self, num_queries, data_train, data_test):
        # create new dataframe that will hold all samples for the experiment
        samples_df = pd.DataFrame(columns=list(data_train.columns.values), dtype=float)

        if num_queries is None:
            num_samples = len(data_train) + len(data_test)
        else:
            num_samples = num_queries

        # record each sample's actual membership. If the sample comes from the training data -> True. If the sample comes
        # from the test data -> False.
        sample_membership = np.empty(num_samples)

        if num_queries is None:
            # if the experiment is executed on all data, simply concatenate the training and test data. We do not need to randomly draw samples
            samples_df = pd.concat([data_train, data_test], ignore_index=True)
            sample_membership[:len(data_train)] = True
            sample_membership[len(data_train):] = False
        else:
            # Otherwise, random samples need to be drawn:
            # half the samples come from the training data, the other half from the test data
            for i in range(num_samples):
                if i % 2 == 0:
                    # choose sample from training data.
                    sample = data_train.sample(random_state=self.rs)
                    sample_membership[i] = True
                    logger.debug('%s taken from training data' % sample.to_numpy())
                else:
                    # choose sample from test data.
                    sample = data_test.sample(random_state=self.rs)
                    sample_membership[i] = False
                    logger.debug('%s taken from test data' % sample.to_numpy())

                samples_df = pd.concat([samples_df, sample], ignore_index=True)

        return samples_df, sample_membership

    def _args_for_parallel_execution(self, threads, samples_df, explainer, model, model_access):
        num_samples = len(samples_df)

        # ceil division. This is equivalent to num_samples / threads (rounded up).
        samples_per_thread = -(num_samples // -threads)

        arg_list = []

        for i in range(threads):
            start_idx = i * samples_per_thread
            end_idx = min((i + 1) * samples_per_thread, num_samples)

            if model_access:
                arg_list.append((explainer, samples_df.iloc[start_idx:end_idx, :], model))
            else:
                arg_list.append((explainer, samples_df.iloc[start_idx:end_idx, :]))

        return arg_list

    @staticmethod
    def _calc_accuracy_precision_recall(actual_membership, inferred_membership):
        samples_in_training_data = np.count_nonzero(actual_membership)
        samples_not_in_training_data = len(actual_membership) - samples_in_training_data

        pred_positives = np.count_nonzero(inferred_membership)

        correct_predictions = np.count_nonzero(np.equal(inferred_membership, actual_membership))
        true_positives = np.count_nonzero(inferred_membership[actual_membership == True])

        accuracy = correct_predictions / len(actual_membership)
        if pred_positives > 0:
            precision = true_positives / pred_positives
        else:
            # If the attack predicted membership for no given sample then precision cannot be calculated
            precision = float("NaN")
        recall = true_positives / samples_in_training_data

        print(f'Accuracy: {accuracy}, precision: {precision}, recall: {recall}')

        return accuracy, precision, recall

    @staticmethod
    def membership_inference_attack_model_access(explainer, samples_df, model):
        """Executes membership inference attack with access to the model

        Abstract method that must be implemented by subclass. Executes the attack against the explainer with access to the
        model. Infers membership for each sample in samples_df. Returns a numpy array with boolean values indicating the
        inferred membership of each given sample. Must be same length as samples_df.

        Parameters
        ----------
        explainer
            The explainer or explanation that will be attacked.
        samples_df
            A pandas dataframe that contains the feature values of all given samples.
        model
            The trained model that is explained by the explainer.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclass.
        """
        raise NotImplementedError

    @staticmethod
    def membership_inference_attack_no_model_access(explainer, samples_df, ignore):
        """Executes membership inference attack without access to the model

        Abstract method that must be implemented by subclass. Executes the attack against the explainer without access to the
        model. Infers membership for each sample in samples_df. Returns a numpy array with boolean values indicating the
        inferred membership of each given sample. Must be same length as samples_df.

        Parameters
        ----------
        explainer
            The explainer or explanation that will be attacked.
        samples_df
            A pandas dataframe that contains the feature values of all given samples.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclass.
        """
        raise NotImplementedError


class TrainingDataExtraction(XaiPrivacyExperiment):
    """
    Executes a training data extraction attack. Some public methods must be implemented by subclass.

    Methods
    -------

    training_data_extraction_experiment(self, num_queries: None or int, model, model_access: bool):
        Executes training data extraction experiment
    training_data_extraction_model_access(explainer, num_queries, feature_format, rng, model):
        Executes training data extraction attack with access to the model
    training_data_extraction_no_model_access(explainer, num_queries, feature_format, rng):
        Executes training data extraction attack without access to the model
    """

    def training_data_extraction_experiment(self, num_queries: None or int, model, model_access: bool, threads=1):
        """Executes training data extraction experiment

        Executes the training data extraction experiment with the dataset that this object was instantiated with. Trains given
        model on dataset and tests precision and recall of the implemented training data extraction attack.
        If model_access is True, the attack method with the parameter "model" is used (the attacker has access to the model).
        Otherwise, the attack method without that parameter is used (the attacker has no access to the model).

        Parameters
        ----------
        num_queries : None or int
            The number of queries allowed for the attacker to extract a sample. If None, the attack can make any number
            of queries to attempt to extract the full dataset.
        model
            The untrained model used in the experiment.
        model_access : bool
            Whether the attack is executed with attacker access to the model or without.
        """
        # stop the time of training model, training explainer, and executing experiment
        start_time = time.time()

        # create pipeline that transforms categorical features to one hot encoding
        model = self._model_pipeline(model)

        # train classifier on dataset
        model = model.fit(self.features, self.labels)

        train_model_time = time.time()

        # train explainer on training data and classifier
        explainer = self.train_explainer(self.data, model)

        train_explainer_time = time.time()

        # generate the feature format information that is available to the attacker
        feature_format = self._generate_feature_info(self.features, self.numeric_features)

        # extract samples using training data extraction attack against the explainer
        if threads > 1:
            arg_list = self._args_for_parallel_execution(threads, explainer, num_queries, feature_format, model,
                                                         model_access)

            if model_access:
                with multiprocessing.Pool(threads) as p:
                    results = p.starmap(self.training_data_extraction_model_access, arg_list)
            else:
                with multiprocessing.Pool(threads) as p:
                    results = p.starmap(self.training_data_extraction_no_model_access, arg_list)

            extracted_samples = pd.concat(results)
        else:
            if model_access:
                extracted_samples = self.training_data_extraction_model_access(explainer, num_queries, feature_format,
                                                                               self.rng, model)
            else:
                extracted_samples = self.training_data_extraction_no_model_access(explainer, num_queries,
                                                                                  feature_format, self.rng)

        # compare the extracted samples to the training data -> number of accurate extractions
        accurate_samples, num_extracted_samples, all_samples = self._compare_data(extracted_samples, self.data,
                                                                                  num_queries)

        end_time = time.time()
        self._print_times(start_time, train_model_time, train_explainer_time, end_time)

        # calculate precision and recall
        return self._calc_precision_recall_tde(accurate_samples, num_extracted_samples, all_samples)

    def _args_for_parallel_execution(self, threads, explainer, num_queries, feature_format, model, model_access):

        avg_queries_per_thread = num_queries / threads

        arg_list = []
        total_queries = 0

        for i in range(threads):
            num_queries_local = int((i + 1) * avg_queries_per_thread) - int(i * avg_queries_per_thread)
            total_queries += num_queries_local

            if i == threads - 1 and total_queries < num_queries:
                num_queries_local += num_queries - total_queries

            if model_access:
                arg_list.append((explainer, num_queries_local, feature_format, self.rng.integers(100000), model))
            else:
                arg_list.append((explainer, num_queries_local, feature_format, self.rng.integers(100000)))

        return arg_list

    @staticmethod
    def _generate_feature_info(features, numeric_features):
        feature_information = []

        features_np = features.to_numpy()

        # Get the minimum and maximum value for all numeric features in the training data.
        # Get the categories for all categorical features.
        for i, feature_name in enumerate(features.columns.values):
            this_feature = {'name': feature_name}

            if feature_name in numeric_features:
                this_feature['isCont'] = True

                this_feature['min'] = np.amin(features_np[:, i])
                this_feature['max'] = np.amax(features_np[:, i])

            else:
                this_feature['isCont'] = False

                this_feature['categories'] = features[feature_name].unique()

            feature_information.append(this_feature)

        return feature_information

    @staticmethod
    def _compare_data(extracted_samples, actual_samples, num_queries: None or int):
        # convert data to numpy so that comparison becomes simpler
        extracted_samples = extracted_samples.to_numpy().astype(float)
        actual_samples = actual_samples.to_numpy().astype(float)

        # If only the features (without the labels) were extracted, then the labels are cut off from the actual_samples array
        # in order to be able to compare the two arrays
        if actual_samples.shape[1] > extracted_samples.shape[1]:
            actual_samples = actual_samples[:, :-1]

        # drop duplicates from the extracted samples and from the actual samples to get accurate precision/recall
        extracted_samples = np.unique(extracted_samples, axis=0)
        actual_samples = np.unique(actual_samples, axis=0)

        # all_samples is the maximum amount of samples that could have been extracted during this attack
        # If num_queries is None, it means the attack attempted to extracted all samples in the training data.
        # Otherwise the attack stopped after num_queries queries.
        if num_queries is None:
            all_samples = len(actual_samples)
        else:
            all_samples = num_queries

        num_extracted_samples = extracted_samples.shape[0]
        num_accurate_samples = 0

        for extracted_sample in extracted_samples:
            logger.debug(f'Extracted sample: {extracted_sample}')

            # Get all indices of the extracted sample in the given training data. features_np == row creates a boolean array
            # with True if the cells match and False otherwise. all(axis=1) returns for each row if all elements in the row
            # are True. np.where returns an array of indices where the boolean array contains the value True.
            close_values = np.isclose(actual_samples, extracted_sample)
            close_rows = close_values.all(axis=1)
            indices_of_sample = np.where(close_rows)[0]

            if indices_of_sample.shape[0] > 0:
                logger.debug(f'Appears in training data at indices {indices_of_sample}')
                num_accurate_samples += 1
            else:
                logger.debug('Does not appear in training data')

        return num_accurate_samples, num_extracted_samples, all_samples

    @staticmethod
    def _calc_precision_recall_tde(accurate_samples, num_extracted_samples, all_samples):
        # Percentage of extracted samples that actually appears within the training data
        if num_extracted_samples > 0:
            precision = accurate_samples / num_extracted_samples
        else:
            # If the attack did not extract a single sample then precision cannot be calculated
            precision = float("NaN")

        recall = accurate_samples / all_samples

        print(f'Number of extracted samples: {num_extracted_samples}')
        print(f'Number of accurate extracted samples: {accurate_samples}')
        print(f'Precision: {precision}, recall: {recall}')

        return precision, recall

    @staticmethod
    def training_data_extraction_model_access(explainer, num_queries, feature_format, rng, model):
        """Executes training data extraction attack with access to the model

        Abstract method that must be implemented by subclass. Executes the attack against the explainer with access to the
        model. Is allowed to make num_queries queries. If num_queries is None, makes as many queries as is necessary to
        attempt to extract the full dataset.
        Returns a dataframe containing all extracted samples.

        Parameters
        ----------
        explainer
            The explainer or explanation that will be attacked.
        num_queries : None or int
            The amount of queries to explainer allowed. If None, any number of queries is allowed.
        feature_format
            A dictionary that contains information for each sample (whether it is numeric or categorical, minimum, maximum,
            the categories)
        rng
            Numpy rng object that can be used for reproducible random decisions.
        model
            The trained model that is explained by the explainer.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclass.
        """
        raise NotImplementedError

    @staticmethod
    def training_data_extraction_no_model_access(explainer, num_queries, feature_format, rng):
        """Executes training data extraction attack without access to the model

        Abstract method that must be implemented by subclass. Executes the attack against the explainer without access to the
        model. Allowed to make num_queries queries to the explainer. If num_queries is None, then there is no limit.
        Returns a dataframe containing all extracted samples.

        Parameters
        ----------
        explainer
            The explainer or explanation that will be attacked.
        num_queries : None or int
            The amount of queries to explainer allowed. If None, any number of queries is allowed.
        feature_format
            A dictionary that contains information for each sample (whether it is numeric or categorical, minimum, maximum,
            the categories)
        rng
            Numpy rng object that can be used for reproducible random decisions.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclass.
        """
        raise NotImplementedError


def run_all_experiments(experiment, dataset_dicts, model_dicts, random_state, num_queries, model_access, threads,
                        is_mem_inf=True, convert_cat_to_str=False, repeat=10):
    results_ = {'dataset': [], 'model': [], 'repetition': [], 'accuracy': [], 'precision': [], 'recall': []}

    results_table = pd.DataFrame(data=results_)

    if type(num_queries) is dict:
        num_queries_dict = num_queries
    else:
        num_queries_dict = {}
        for dataset_dict in dataset_dicts:
            num_queries_dict[dataset_dict['name']] = num_queries

    rng = np.random.default_rng(random_state)

    for dataset_dict in dataset_dicts:
        # DiCE needs categorical features to be strings
        if convert_cat_to_str:
            cat_features = dataset_dict['dataset'].columns.difference(dataset_dict['num'] + [dataset_dict['outcome']])
            for col in cat_features:
                dataset_dict['dataset'][col] = dataset_dict['dataset'][col].astype(str)

        for model_dict in model_dicts:
            dataset_name = dataset_dict['name']
            model_name = model_dict['name']

            for i in range(repeat):
                print(f'dataset: {dataset_name}, model: {model_name} (repetition {i})')

                EXP = experiment(dataset_dict['dataset'], dataset_dict['num'], dataset_dict['outcome'],
                                 random_state=rng.integers(1000))

                if is_mem_inf:
                    accuracy, precision, recall = EXP.membership_inference_experiment(
                        num_queries=num_queries_dict[dataset_name], model=model_dict['model'](random_state=rng.integers(1000)),
                        model_access=model_access, threads=threads)
                else:
                    accuracy = -1.0
                    precision, recall = EXP.training_data_extraction_experiment(num_queries=num_queries_dict[dataset_name],
                                                                                model=model_dict['model'](
                                                                                    random_state=rng.integers(1000)),
                                                                                model_access=model_access, threads=threads)
                results_table.loc[len(results_table.index)] = [dataset_name, model_name, i, accuracy, precision, recall]

                results_table.to_csv('cache.csv', index=False, na_rep='NaN', float_format='%.3f')

    return results_table
