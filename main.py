import subprocess

def run_project():
    # Run module1.py as a script
    subprocess.run(["python", "obtain_match_data.py"])
    print("Success 1")
    subprocess.run(["python", "match_outcome_randomizer.py"])
    print("Success 2")
    subprocess.run(["python", "outcome_predictor.py"])
    print("Success 3")
    subprocess.run(["python", "draft_transformer_model.py"])
    print("Success 4")
    subprocess.run(["python", "predict_outcome_of_generated_draft.py"])
    print("Success 5")


if __name__ == "__main__":
    run_project()