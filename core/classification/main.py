import logging

from classification.simple_logistic_regression.logistic_regression_star import (
    LogisticRegressionModelStar,
)

logger = logging.getLogger(__name__)


def run_classification_models(data_folder):
    """
    Run the classification models
    """
    logger.info("Running classification models")

    LogisticRegressionModelStar(data_folder).compute_test_accuracy()

    logger.info("Classification models done")
