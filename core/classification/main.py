import logging

from classification.simple_logistic_regression.logistic_regression_star import (
    LogisticRegressionModelStar,
)
from classification.neural_networks.one_layer_model import OneLayerModel
from classification.neural_networks.multi_layer_model import MultiLayerModel
from classification.neural_networks.perceptron import PerceptronClassifier

logger = logging.getLogger(__name__)


def run_classification_models(data_folder):
    """
    Run the classification models
    """
    logger.info("Running classification models")

    LogisticRegressionModelStar(data_folder).compute_test_accuracy()

    one_layer_perceptron = PerceptronClassifier(
        data_folder, model_structure=OneLayerModel
    )
    one_layer_perceptron.compute_test_accuracy()
    one_layer_perceptron.compute_holdout_accuracy()

    PerceptronClassifier(
        data_folder, model_structure=MultiLayerModel
    ).compute_test_accuracy()

    logger.info("Classification models done")
