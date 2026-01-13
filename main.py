#!/usr/bin/env python
"""Main entry point."""
import sys
sys.path.append('worldcup_pkg')
from worldcup_pkg import load_worldcup_data, build_match_features, train_models

# Full pipeline
wc = load_worldcup_data()
intl = load_international_data()
features = build_match_features(wc, intl)
models = train_models(features[FEATURE_COLS], features['home_win'])
print("Models trained!")
