import os
import pandas as pd

from constants.columns import DatasetColumns
from constants.seed import RANDOM_SEED


class Preprocessor:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.raw_dataset = pd.read_csv(os.path.join(data_folder, "raw/raw_dataset.csv"))


        # Create the preprocessed folder it it doesn't exist
        os.makedirs(os.path.join(data_folder, "preprocessed"), exist_ok=True)

    def create_holdout_set(self):
        """
        Create and export holdout set from our dataset
        """
        # Split the dataset in two
        self.holdout_dataset = self.raw_dataset.sample(frac=0.2, random_state=RANDOM_SEED)
        self.raw_dataset_without_holdout = self.raw_dataset.drop(self.holdout_dataset.index)

        # Export the holdout set
        self.holdout_dataset.to_csv(
            os.path.join(self.data_folder, "preprocessed/holdout_dataset.csv"),
            index=False,
        )

    def select_covariates(self, dataset):
        return dataset[
            [col.value for col in DatasetColumns]
        ]

    def export_preprocessed_dataset(self, dataset, filename):
        dataset.to_csv(
            os.path.join(self.data_folder, "preprocessed", filename),
            index=False,
        )

    @staticmethod
    def one_hot_encode_categorical_variables(dataframe: pd.DataFrame, columns: list):
        """
        One hot encode the categorical variables
        """
        dataframe = pd.get_dummies(dataframe, columns=columns)
        return dataframe

    def remove_outliers(self, dataset):
        """
        We have outliers in the data that we need to remove. They are characterized by having a value in the covariates that are inferior to -9000.
        """
        numerical_columns = dataset.select_dtypes(
            include=["number"]
        ).columns
        dataset[
            numerical_columns
        ] = dataset[numerical_columns].mask(
            dataset[numerical_columns] < -9000, pd.NA
        )

        dataset = dataset.dropna(
            subset=numerical_columns
        )
        return dataset

    def run_preprocessing(self):
        """
        Run the preprocessing pipeline
        """
        self.create_holdout_set()

        # Run preprocessing on the main dataset

        dataset_not_holdout = self.select_covariates(self.raw_dataset_without_holdout)
        dataset_not_holdout = self.remove_outliers(dataset_not_holdout)
        self.preprocessed_dataset = Preprocessor.one_hot_encode_categorical_variables(
            dataset_not_holdout, [DatasetColumns.CLASS.value]
        )

        self.export_preprocessed_dataset(self.preprocessed_dataset, "preprocessed_dataset.csv")

        # Run preprocessing on the holdout dataset
        dataset_holdout = self.select_covariates(self.holdout_dataset)
        dataset_holdout = self.remove_outliers(dataset_holdout)
        self.preprocessed_holdout_dataset = Preprocessor.one_hot_encode_categorical_variables(
            dataset_holdout, [DatasetColumns.CLASS.value]
        )

        self.export_preprocessed_dataset(self.preprocessed_holdout_dataset, "preprocessed_holdout_dataset.csv")