# plotting helpers

import matplotlib.pyplot as plt
import seaborn as sns
from .models import FEATURE_COLS

def plot_predictions(probs):
    """Bar plot of winning probabilities."""
    teams = list(probs.keys())
    probs_list = list(probs.values())
    plt.figure(figsize=(10, 6))
    sns.barplot(x=probs_list, y=teams)
    plt.xlabel('Win Probability')
    plt.title('2026 World Cup Winner Probabilities')
    plt.tight_layout()
    plt.savefig('../reports/figures/winner_probs.png')
    plt.show()

def plot_feature_importance(model):
    """Feature importance for tree models."""
    if hasattr(model, 'feature_importances_'):
        feat_imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        feat_imp.sort_values().plot(kind='barh')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('../reports/figures/feat_importance.png')
        plt.show()
