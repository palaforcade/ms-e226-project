import os
import pandas as pd

from constants.columns import DatasetColumns


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
        holdout_dataset = self.raw_dataset.sample(frac=0.2, random_state=42)
        self.raw_dataset_without_holdout = self.raw_dataset.drop(holdout_dataset.index)

        # Export the holdout set
        holdout_dataset.to_csv(
            os.path.join(self.data_folder, "preprocessed/holdout_dataset.csv"),
            index=False,
        )

    def select_covariates(self):
        self.preprocessed_dataset = self.raw_dataset_without_holdout[
            [col.value for col in DatasetColumns]
        ]

    def export_preprocessed_dataset(self):
        self.preprocessed_dataset.to_csv(
            os.path.join(self.data_folder, "preprocessed/preprocessed_dataset.csv"),
            index=False,
        )

    def run_preprocessing(self):
        """
        Run the preprocessing pipeline
        """

        self.create_holdout_set()

        self.select_covariates()

        self.export_preprocessed_dataset()
