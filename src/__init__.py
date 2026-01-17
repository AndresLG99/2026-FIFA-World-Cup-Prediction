# Make package importable
from .data_loading import load_worldcup_data, load_international_data, load_player_stats
from .features import build_match_features
from .models import train_models, predict_match
from .simulate import simulate_world_cup
from .viz import plot_predictions, plot_feature_importance

__all__ = [
    "load_worldcup_data", "load_international_data", "load_player_stats",
    "build_match_features", "train_models", "predict_match",
    "simulate_world_cup", "plot_predictions", "plot_feature_importance"
]
