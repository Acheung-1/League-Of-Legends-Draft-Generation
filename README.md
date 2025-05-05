# League-Of-Legends-Draft-Generation

Note: Models were create with match data with the latest patch of 15.8
      Models have been trained on 14000 unique matches each, collected from the past 100 matches of the top 300 challenger players 
      main.py is used to collect/preprocess match data and train models
      flask_website_champion_select.py serves a webpage for interacting with ML draft models

### main.py
- Will need access to riot_api_key, attained by going to https://developer.riotgames.com/apis
  - Log in to Riot Account and request the 24 hour key
  - Create an .env file and paste key (EX: riot_api_key="KEY_HERE")
- Main script that collects and preprocesses match data and train them in outcome predictor and transformer model
- Handles configuration for different regions (NA, KR, EUW)
- Executes a 4-step process:
  1. obtain_match_data.py – Fetches match data using Riot API
  2. match_outcome_randomizer.py – Randomizes match data for training
  3. outcome_predictor_model.py – Trains an outcome predictor model
  4. draft_transformer_model.py – Trains a transformer model for draft generation

### flask_website_champion_select.py
- Flask web application for the champion draft interface, runs on local host
- Provides REST endpoints for:
  - Home page with champion selection interface
  - Draft generation API endpoint
- Implements champion draft generation using trained models
- Select Enemy team draft on left side
- Choose region-specific model (KR, NA, EUW)
- Click "Generate Counter Draft", Counter Draft will populate on the right side
- Includes win probability prediction for generated drafts from the outcome predictor model that is trained on the same 14000 matches

### obtain_match_data.py
- Handles data collection from Riot Games API
- Key features:
  - Fetches leaderboard data for top players
  - Collects match history from high-ranked players
  - Processes and stores match data
  - Supports multiple regions and leagues (Challenger, Grandmaster, Master)

### match_outcome_randomizer.py
- Processes and stores match data
- Randomizes order of team compositions and stores match outcome
  - [team 1, team 2]
  - team 2 wins: 1, team 2 loses: 0

### outcome_predictor_model.py
- Implements a logistic regression model to predict match outcomes
- Key features:
  - One-hot encoding for champion compositions
  - Binary classification (win/loss prediction)

### draft_transformer_model.py
- Implements a transformer-based model for generating team compositions
- Key features:
  - RoleAwareTransformer: Custom transformer architecture with role embeddings
  - Multiple sampling strategies (top-k, top-p, temperature)

### folders
- models: contains the follow for each region (NA, KR, EUW)
  - binary outcome predictory model (trained on 14000 matches)
  - draft transformer model (trained on 14000 matches)
- templates: contains index.html
- static/css: contains styles.css