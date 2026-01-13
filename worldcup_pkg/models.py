# train / evaluate models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import joblib
from .config import TEST_YEAR

FEATURE_COLS = ['home_win_rate', 'away_win_rate', 'home_goal_diff', 'away_goal_diff', 'form_diff', 'strength_diff']


def train_models(X_train, y_train, X_val=None, y_val=None):
    """Train baseline models."""
    models = {
        'logreg': LogisticRegression(random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgb': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

        if X_val is not None:
            probs = model.predict_proba(X_val)[:, 1]
            print(f"{name} log loss: {log_loss(y_val, probs):.3f}")

    return trained


def predict_match(model, match_features):
    """Predict P(home win) for a match."""
    X = match_features[FEATURE_COLS].values.reshape(1, -1)
    return model.predict_proba(X)[:, 1][0]
