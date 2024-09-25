import os

from sklearn.compose import ColumnTransformer

from pipelines import build_pipeline

TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # Mark the test root directory


# <DEFINE YOUR TESTS FOR PIPELINE>
class TestPipelines:
    def test_build_pipeline(self):
        pipeline = build_pipeline()
        assert type(pipeline) == ColumnTransformer
        assert len(pipeline.transformers[0][1].steps) == 2
        assert len(pipeline.transformers[1][1].steps) == 2
