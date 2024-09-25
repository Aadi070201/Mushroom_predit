from sklearn.base import is_classifier

from models import build_model


class TestModels:
    def test_build_model(self):
        model = build_model()
        assert is_classifier(model)
