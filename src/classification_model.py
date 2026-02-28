"""
Classification Model Module
CRISP-DM Phase 4: Modeling - Cancellation Prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import os


class CancellationClassifier:
    """Classification model for booking cancellation prediction."""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_names = None

    def _create_model(self):
        """Create the base model."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, tune_hyperparams: bool = False
    ):
        """Train the classification model."""
        self.feature_names = list(X_train.columns)
        self.model = self._create_model()

        if tune_hyperparams:
            self._tune_hyperparameters(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

        return self

    def _tune_hyperparameters(self, X_train, y_train):
        """Tune hyperparameters using GridSearchCV."""
        if self.model_type == "xgboost":
            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
            }
        else:
            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [5, 10],
                "min_samples_split": [2, 5],
            }

        grid_search = GridSearchCV(
            self.model, param_grid, cv=3, scoring="f1", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cancellation (0/1)."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cancellation probability."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred))
        print(f"\nROC-AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        importance = self.model.feature_importances_
        return pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

    def save(self, filepath: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "feature_names": self.feature_names,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CancellationClassifier":
        """Load model from disk."""
        data = joblib.load(filepath)
        instance = cls(model_type=data["model_type"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        return instance


if __name__ == "__main__":
    # Example usage
    from preprocessing import (
        load_data,
        clean_data,
        engineer_features,
        prepare_classification_data,
    )

    # Load and prepare data
    df = load_data("../data/raw/hotel_bookings.csv")
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, features = prepare_classification_data(df)

    # Train model
    clf = CancellationClassifier(model_type="xgboost")
    clf.train(X_train, y_train)

    # Evaluate
    metrics = clf.evaluate(X_test, y_test)

    # Feature importance
    print("\nTop 10 Features:")
    print(clf.get_feature_importance().head(10))

    # Save model
    clf.save("../models/cancellation_model.pkl")
