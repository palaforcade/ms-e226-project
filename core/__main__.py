import argparse
import logging

from constants.task import Task
from preprocessing.main import Preprocessor

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
