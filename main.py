from dotenv import load_dotenv
import os

from obtain_match_data import main as obtain_data_main
from match_outcome_randomizer import main as randomizer_main
from outcome_predictor import main as predictor_main
from draft_transformer_model import main as transformer_main
# from predict_outcome_of_generated_draft import main as generated_outcome_main

def generate_file_paths(region_for_leaderboard):
    return {
        "leaderboard_df": f"{region_for_leaderboard}_leaderboard_df.txt",
        "match_ids": f"{region_for_leaderboard}_all_match_ids.txt",
        "processed_puuids": f"{region_for_leaderboard}_processed_player_puuids.txt",
        "processed_matches": f"{region_for_leaderboard}_processed_match_ids.txt",
        "training_data": f"{region_for_leaderboard}_training_data.txt",
        "randomized_training_data": f"{region_for_leaderboard}_randomized_training_data.txt",
        "binary_predictor_model": f"{region_for_leaderboard}_binary_predictor_model.pt",
        "transformer_model": f"{region_for_leaderboard}_transformer_model.pt",
        "generated_draft_predicted_outcomes": f"{region_for_leaderboard}_generated_draft_predicted_outcomes.txt",
        "formatted_training_data": f"{region_for_leaderboard}_formatted_training_data.txt",
        "x_randomized_team_comp": f"{region_for_leaderboard}_x_randomized_team_comp.txt",
        "y_randomized_outcomes": f"{region_for_leaderboard}_y_randomized_outcomoes.txt",
    }

def run_project():
    load_dotenv()
    riot_api_key = os.getenv("riot_api_key")
    if riot_api_key is None:
        raise ValueError("Missing riot_api_key in .env file")

    ""
    # regions = [["na1","americas"],["kr","asia"],["euw1","europe"]]
    regions = [["na1","americas"]]

    for region_for_leaderboard, region_country in regions:
        # Shared config
        config = {
            "riot_api_key":         riot_api_key,
            "region_country":       region_country,
            "region_leaderboard":   region_for_leaderboard,
            "player_count":         5,
            "queue_type":           "RANKED_SOLO_5x5",
            "leagues":              ["challengerleagues"],
            "match_type":           "ranked",
            "number_matches":       5,
            "file_paths":           generate_file_paths(region_for_leaderboard)
        }


        # Step 1: Fetch match data
        obtain_data_main(riot_api_key=config["riot_api_key"],
                         leaderboard_df_file_path=config["file_paths"]["leaderboard_df"],
                         all_match_ids_file_path=config["file_paths"]["match_ids"],
                         processed_player_puuids_file_path=config["file_paths"]["processed_puuids"],
                         processed_match_ids_file_path=config["file_paths"]["processed_matches"],
                         data_file_path=config["file_paths"]["training_data"],
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
        randomizer_main(unformatted_training_data=config["file_paths"]["training_data"], 
                        formatted_training_data=config["file_paths"]["formatted_training_data"],
                        randomized_team_comp=config["file_paths"]["x_randomized_team_comp"],
                        randomized_outcomes=config["file_paths"]["y_randomized_outcomes"],
        )
        print("Success 2")

        # # Step 3: Train predictor
        # predictor_main(data_file="randomized_training_data.txt", model_output=config["binary_predictor_model"])
        # print("Success 3")

        # # Step 4: Train transformer
        # transformer_main(data_file="randomized_training_data.txt", model_output=config["transformer_model"])
        # print("Success 4")

        # # Step 5: Predict outcomes with generated drafts
        # generated_outcome_main(
        #     transformer_model=config["transformer_model"],
        #     predictor_model=config["binary_predictor_model"],
        #     output_file=config["generated_draft_predicted_outcomes"]"
        # )
        # print("Success 5")

if __name__ == "__main__":
    run_project()