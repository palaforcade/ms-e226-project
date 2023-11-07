import argparse
import logging

import config.logging
from constants.task import Task
from preprocessing.main import Preprocessor
from exploration.plots import DatasetExplorer
from regression.main import run_regression_models
from classification.main import run_classification_models

logger = logging.getLogger(__name__)


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse the arguments for the task to be done"
    )

    parser.add_argument(
        "--task",
        type=Task,
        help="The task to execute",
    )

    parser.add_argument(
        "--data-folder",
        type=str,
        help="The path to the data folder",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Running the task : {args.task}")

    if args.task == Task.PREPROCESSING:
        logger.info("Running preprocessing task")
        Preprocessor(args.data_folder).run_preprocessing()
        logger.info("Preprocessing task done")

    if args.task == Task.EXPLORATION:
        logger.info("Running exploration task")
        DatasetExplorer(args.data_folder).run_exploration()
        logger.info("Exploration task done")

    if args.task == Task.REGRESSION:
        logger.info("Running regression task")
        run_regression_models(args.data_folder)
        logger.info("Regression task done")

    if args.task == Task.CLASSIFICATION:
        logger.info("Running classification task")
        run_classification_models(args.data_folder)
        logger.info("Classification task done")
