from flask import Flask, render_template, request, jsonify
import torch
from outcome_predictor_model import LogisticRegressionModel, ids_to_one_hot
from draft_transformer_model import RoleAwareTransformer
from champion_dictionary import Champion_to_Id, Id_to_Champion

app = Flask(__name__)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the draft generator model
vocab_size = 951  # From your existing code
draft_generator = RoleAwareTransformer(vocab_size=vocab_size, embed_dim=128)
draft_generator.load_state_dict(torch.load('draft_predictor_model_na_challenger.pt', map_location=device))
draft_generator.to(device)
draft_generator.eval()

# Initialize and load the outcome predictor model
outcome_predictor = LogisticRegressionModel(input_dim=1710)  # 171 champions * 10
outcome_predictor.load_state_dict(torch.load('outcome_predictor_model_na_challenger.pt', map_location=device))
outcome_predictor.to(device)
outcome_predictor.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_draft', methods=['POST'])
def generate_draft():
    data = request.get_json()
    enemy_team = data.get('enemy_team', [])  # List of champion names
    
    # Convert champion names to IDs
    enemy_ids = [Champion_to_Id[champ] for champ in enemy_team]
    seed = enemy_ids + [0,0,0,0,0]  # Add placeholders for our team
    
    # Generate drafts using different methods
    drafts = []
    sampling_methods = [
        ("top_k", {"k": 3}),
        ("top_p", {"p": 0.9}),
        ("temperature", {"temperature": 0.8})
    ]
    
    for method, params in sampling_methods:
        generated_draft = generate_team(
            draft_generator,
            seed,
            vocab_size=vocab_size,
            device=device,
            sampling_method=method,
            **params
        )
        
        # Convert IDs back to champion names
        team1 = [Id_to_Champion[id] for id in generated_draft[:5]]
        team2 = [Id_to_Champion[id] for id in generated_draft[5:]]
        
        # Get win probability
        win_prob = predict_match_outcome(outcome_predictor, generated_draft[:5], generated_draft[5:], device)
        
        drafts.append({
            'method': method,
            'enemy_team': team1,
            'generated_team': team2,
            'win_probability': float(win_prob)
        })
    
    return jsonify({'drafts': drafts})

if __name__ == '__main__':
    app.run(debug=True)