import os
import pandas as pd
from uuid import uuid4 as uui
from datetime import datetime as dt
import math
from matplotlib import pyplot as plt

from constants.columns import DatasetColumns


class DatasetExplorer:
    def __init__(self, data_folder) -> None:
        self.data_folder = data_folder
        self.dataset = pd.read_csv(
            os.path.join(data_folder, "preprocessed/preprocessed_dataset.csv")
        )

    def plot_redshift_function_of_declination(self):
        plt.plot(
            self.dataset[DatasetColumns.DECLINATION_J2000.value],
            self.dataset[DatasetColumns.REDSHIFT.value],
            "o",
            markersize=0.15,
        )

        # Legend the plot
        plt.xlabel("Declination (degrees)")
        plt.ylabel("Redshift")
        plt.title("Redshift as a function of declination")

    def plot_redshift_function_of_column_set(self, column_set: list[DatasetColumns]):
        col_count = int(math.ceil(math.sqrt(len(column_set))))
        row_count = int(math.ceil(len(column_set) / col_count))

        fig, axs = plt.subplots(row_count, col_count, figsize=(15, 10))

        for i, col in enumerate(column_set):
            row_number = int(i / col_count)
            col_number = i % col_count

            axs[row_number, col_number].plot(
                self.dataset[col.value],
                self.dataset[DatasetColumns.REDSHIFT.value],
                "o",
                markersize=0.15,
            )

            axs[row_number, col_number].set_title(
                "Redshift as a function of " + col.name
            )
            axs[row_number, col_number].set_xlabel(col.name)
            axs[row_number, col_number].set_ylabel("Redshift")

        os.makedirs(os.path.join(self.data_folder, "exploration/plots"), exist_ok=True)
        fig.savefig(
            os.path.join(
                self.data_folder,
                f"exploration/plots/{uui()}.png",
            )
        )

    def run_exploration(self):
        self.plot_redshift_function_of_declination()
        self.plot_redshift_function_of_column_set(
            [
                DatasetColumns.MAGNITUDE_FIT_U,
                DatasetColumns.MAGNITUDE_FIT_G,
                DatasetColumns.MAGNITUDE_FIT_R,
                DatasetColumns.MAGNITUDE_FIT_I,
                DatasetColumns.MAGNITUDE_FIT_Z,
            ]
        )

        self.plot_redshift_function_of_column_set(
            [
                DatasetColumns.PETROSIAN_RADIUS_U,
                DatasetColumns.PETROSIAN_RADIUS_G,
                DatasetColumns.PETROSIAN_RADIUS_I,
                DatasetColumns.PETROSIAN_RADIUS_R,
                DatasetColumns.PETROSIAN_RADIUS_Z,
            ]
        )

        self.plot_redshift_function_of_column_set(
            [
                DatasetColumns.PETROSIAN_FLUX_U,
                DatasetColumns.PETROSIAN_FLUX_G,
                DatasetColumns.PETROSIAN_FLUX_I,
                DatasetColumns.PETROSIAN_FLUX_R,
                DatasetColumns.PETROSIAN_FLUX_Z,
            ]
        )

        self.plot_redshift_function_of_column_set(
            [
                DatasetColumns.PETROSIAN_HALF_LIGHT_RADIUS_U,
                DatasetColumns.PETROSIAN_HALF_LIGHT_RADIUS_G,
                DatasetColumns.PETROSIAN_HALF_LIGHT_RADIUS_I,
                DatasetColumns.PETROSIAN_HALF_LIGHT_RADIUS_R,
                DatasetColumns.PETROSIAN_HALF_LIGHT_RADIUS_Z,
            ]
        )

        self.plot_redshift_function_of_column_set(
            [
                DatasetColumns.PSF_MAGNITUDE_U,
                DatasetColumns.PSF_MAGNITUDE_G,
                DatasetColumns.PSF_MAGNITUDE_I,
                DatasetColumns.PSF_MAGNITUDE_R,
                DatasetColumns.PSF_MAGNITUDE_Z,
            ]
        )

        plt.show()
