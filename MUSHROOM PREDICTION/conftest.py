import os

import pandas as pd
import pytest

from pipelines import apply_pipeline, build_pipeline, train_pipeline

TEST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test"
)  # Mark the test root directory


# Pytest loader requires a conftest with (empty) initialization
def __init__():
    pass


# Context to retain objects passed between steps in test scenarios
class Context:
    pass


# Fixtures that define prototype objects for testing purposes
@pytest.fixture(scope="session")
def context():
    return Context()


@pytest.fixture
def raw_data_set():
    test_csv_path = os.path.join(TEST_DIR, "dataset.csv")
    return pd.read_csv(test_csv_path, sep=";")


@pytest.fixture
def clean_data_set():
    test_csv_path = os.path.join(TEST_DIR, "dataset.csv")
    raw_data = pd.read_csv(test_csv_path, sep=";")
    pipeline = build_pipeline()
    train_pipeline(pipeline, raw_data)
    return apply_pipeline(pipeline, raw_data)
