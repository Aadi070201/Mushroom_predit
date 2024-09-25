import os

from pytest_bdd import given, scenario, then, when

from data_loader import split_train_test
from models import load_model
from pipelines import apply_pipeline, load_pipeline

TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # Mark the test root directory


@scenario("predict_edibility.feature", "Pre-process a data set")
def test_pre_process_data_set():
    pass


@given("a raw data set")
def load_raw_data_sets(context, raw_data_set):
    context.train_set, context.test_set = split_train_test(raw_data_set)


@when("I pre-process the raw data set for analysis")
def pre_process_data_sets(context):
    context.pipeline = load_pipeline(TEST_DIR)
    context.X_train, context.y_train = apply_pipeline(
        context.pipeline, context.train_set
    )
    context.X_test, context.y_test = apply_pipeline(context.pipeline, context.test_set)


@then("I obtain a pre-processed data set")
def obtain_pre_processed_data_sets(context):
    assert context.X_train.shape == (24, 26)
    assert context.y_train.shape == (24,)
    assert context.X_test.shape == (6, 26)
    assert context.y_test.shape == (6,)


@scenario("predict_edibility.feature", "Obtain predictions for new data")
def test_obtain_predictions():
    pass


@given("a trained model")
def trained_model(context):
    # Load a pre-trained model from the test directory
    context.model = load_model(TEST_DIR)


@when("I use the trained model for prediction on new data")
def predict(context, clean_data_set):
    X_pred, _ = clean_data_set  # Unpack
    context.y_pred = context.model.predict(X_pred)


@then("I obtain estimated edibility")
def obtain_estimates(context):
    assert context.y_pred.shape == (30,)
