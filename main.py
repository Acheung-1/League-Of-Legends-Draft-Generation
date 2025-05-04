import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import ast
from champion_dictionary import Id_to_Champion, Champion_to_Id, Id_to_Consecutive_Id

from dotenv import load_dotenv
import os

from obtain_match_data import main as obtain_data_main
from match_outcome_randomizer import main as randomizer_main

from outcome_predictor_model import main as predictor_main
from outcome_predictor_model import LogisticRegressionModel

from draft_transformer_model import main as transformer_main
from draft_transformer_model import RoleAwareTransformer

def generate_text_file_paths(region_for_leaderboard):
    return {
        "leaderboard_df": f"{region_for_leaderboard}_1_leaderboard_df.txt",
        "match_ids": f"{region_for_leaderboard}_2_all_match_ids.txt",
        "processed_puuids": f"{region_for_leaderboard}_3_processed_player_puuids.txt",
        "processed_matches": f"{region_for_leaderboard}_4_processed_match_ids.txt",
        "unformatted_training_data": f"{region_for_leaderboard}_5_unformatted_training_data.txt",
        "randomized_training_data": f"{region_for_leaderboard}_6_randomized_training_data.txt",
        "generated_draft_predicted_outcomes": f"{region_for_leaderboard}_7_generated_draft_predicted_outcomes.txt",
        "formatted_training_data": f"{region_for_leaderboard}_8_formatted_training_data.txt",
        "x_randomized_team_comp": f"{region_for_leaderboard}_9_x_randomized_team_comp.txt",
        "y_randomized_outcomes": f"{region_for_leaderboard}_10_y_randomized_outcomes.txt",
    }

def generate_models_file_paths(region_for_leaderboard,leagues):
    league_text = ""
    for league in leagues:
        league_name = league.replace("leagues", "")
        league_text += f"_{league_name}"
    
    return {
        "outcome_predictor_model": f"{region_for_leaderboard}{league_text}_model_1_binary_predictor.pt",
        "transformer_model": f"{region_for_leaderboard}{league_text}_model_2_transformer.pt",
    }


def run_project():
    load_dotenv()
    riot_api_key = os.getenv("riot_api_key")
    if riot_api_key is None:
        raise ValueError("Missing riot_api_key in .env file")

    ""

    # Choose which regions to generate a model from: below is an example of multiple regions
    # Default is north american region
    # regions = [["na1","americas"],["kr","asia"],["euw1","europe"]]
    regions = [["na1","americas"]]
    leagues = ["challengerleagues"]

    for region_for_leaderboard, region_country in regions:
        config = {
            "riot_api_key":         riot_api_key,
            "region_country":       region_country,
            "region_leaderboard":   region_for_leaderboard,
            "player_count":         5,
            "queue_type":           "RANKED_SOLO_5x5",
            "leagues":              leagues,
            "match_type":           "ranked",
            "number_matches":       5,
            "text_file_paths":      generate_text_file_paths(region_for_leaderboard),
            "model_file_paths":     generate_models_file_paths(region_for_leaderboard,leagues)
        }

        # Step 1: Fetch match data
        obtain_data_main(riot_api_key=config["riot_api_key"],
                         leaderboard_df_file_path=config["text_file_paths"]["leaderboard_df"],
                         all_match_ids_file_path=config["text_file_paths"]["match_ids"],
                         processed_player_puuids_file_path=config["text_file_paths"]["processed_puuids"],
                         processed_match_ids_file_path=config["text_file_paths"]["processed_matches"],
                         data_file_path=config["text_file_paths"]["unformatted_training_data"],
                         region_leaderboard=config["region_leaderboard"],
                         region_country=config["region_country"],
                         player_count=config["player_count"],
                         queue_type=config["queue_type"],
                         leagues=config["leagues"],
                         match_type=config["match_type"],
                         number_matches=config["number_matches"]
        )
        print("Success 1")

        # Step 2: Run randomizer
        randomizer_main(unformatted_training_data=config["text_file_paths"]["unformatted_training_data"], 
                        formatted_training_data=config["text_file_paths"]["formatted_training_data"],
                        randomized_team_comp=config["text_file_paths"]["x_randomized_team_comp"],
                        randomized_outcomes=config["text_file_paths"]["y_randomized_outcomes"],
        )
        print("Success 2")

        # Step 3: Train predictor
        predictor_main(randomized_team_comp=config["text_file_paths"]["x_randomized_team_comp"],
                       randomized_outcomes=config["text_file_paths"]["y_randomized_outcomes"],
                       outcome_predictor_model=config["model_file_paths"]["outcome_predictor_model"]
        )
        print("Success 3")

        # Example of how to load and use the outcome predictor model
        # device = "cpu"
        # input_dim = 1710
        # model = LogisticRegressionModel(input_dim)
        # model.load_state_dict(torch.load(config["model_file_paths"]["binary_predictor_model"]))
        # model.to(device)
        # model.eval()
        # # Predict for new champion IDs
        # ids = [266, 103, 84, 12, 34, 875, 238, 67, 555, 142]
        # prediction = model.predict_outcome(ids, device=device)
        # print(f"Predicted outcome: {prediction}")

        # Step 4: Train transformer
        transformer_main(formatted_training_data=config["text_file_paths"]["formatted_training_data"], 
                         transformer_draft_generator_model=config["model_file_paths"]["transformer_model"]
        )
        print("Success 4")

        # Example of how how to load and use the draft transformer model
        # model = RoleAwareTransformer(vocab_size=vocab_size, embed_dim=embed_dim)  # Create the model first
        # model.load_state_dict(torch.load('draft_predictor_model_na_challenger.pt', map_location=device))  # Load weights
        # model.to(device)
        # model.eval()

        # # Example usage
        # ids = [150, 80, 268, 110, 235, 897, 234, 4, 81, 117]  # patch + losing team (5 picks total)
        # generate_multiple_drafts(model,ids,device=device,top_k_drafts=3, top_p_drafts=3, temp_drafts=1)

if __name__ == "__main__":
    run_project()