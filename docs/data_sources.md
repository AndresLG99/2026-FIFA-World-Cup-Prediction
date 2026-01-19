# Data Sources Bibliography

This project combines international match results, a 2026 World Cup fixtures template, historical World Cup tables, and club-season player statistics to build expected-goals models and run tournament simulations.

## 1) International match results (1872–present)

- **International Football Results: Daily Updates** (Kaggle, patateriedata).
-  Used as the main match-history dataset for training and for building match-history features (Elo and rolling form).  
-  Project file: `data/raw/main/all_matches.csv`.

## 2) Historical World Cup database (1930–2022)

- **Fjelstul World Cup Database** (GitHub repository `jfjelstul/worldcup`, `data-csv/`).
-  Used for World Cup historical tables (matches, teams, squads, goals, and player appearances).  
-  Project files (stored under `data/raw/players_data-2011_2022/`): `matches.csv`, `teams.csv`, `squads.csv`, `goals.csv`, `player_appearances.csv`.

## 3) Club-season player statistics (Top 5 European leagues)

These datasets are used to build minutes-weighted team player features that power the PLAYER model. 

- **All Football Players Stats in Top 5 Leagues 23/24** (Kaggle, orkunaktas).   
  Project file: `data/raw/players_data-2023_2026/player_data-2023_2024.csv`. 

- **Football Players Stats (2024–2025)** (Kaggle, hubertsidorowicz).   
  Project file: `data/raw/players_data-2023_2026/players_data-2024_2025.csv`. 

- **Football Players Stats (2025–2026)** (Kaggle, hubertsidorowicz).   
  Project file: `data/raw/players_data-2023_2026/players_data-2025_2026.csv`.

## 4) 2026 World Cup fixtures schedule

- **2026 World Cup Schedule – USA, Canada and Mexico** (Roadtrips).   
  Used as the source for the project’s 2026 match schedule template.   
  Project file: `data/raw/fixtures/all_matches_WC_2026.csv`. 

## 5) Project-specific mapping tables

These tables were created/curated for this project to standardize names and connect player nationality codes to national team names.

- `data/raw/main/countries_names.csv` (team name standardization).  
- `data/raw/main/nation_code_map.csv` (player nationcode → national team name). 