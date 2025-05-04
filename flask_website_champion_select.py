from flask import Flask, render_template, jsonify, request
from champion_dictionary import Champion_to_Id, Id_to_Champion, Consecutive_Id_to_Id, Id_to_Consecutive_Id
import torch
from draft_transformer_model import RoleAwareTransformer, generate_team
from outcome_predictor_model import LogisticRegressionModel

app = Flask(__name__, static_url_path='/static')

def generate_multiple_drafts(model, riot_champ_ids, device, top_k_drafts=3, top_p_drafts=3, temp_drafts=1):
    '''
    DESCRIPTION:
        Generate multiple winning team drafts using various stochastic sampling strategies for comparison
    
    INPUTS:
        model (nn.Module):            Trained transformer model
        riot_champ_ids (list[int]):   Full draft of 10 champion IDs using Riot champ ids(first 5 are losing team)
        device (str):                 'cuda' or 'cpu'
        top_k_drafts (int):           Number of drafts to generate using top-k sampling
        top_p_drafts (int):           Number of drafts to generate using top-p sampling
        temp_drafts (int):            Number of drafts to generate using temperature sampling
    
    OUTPUTS:
        drafts (array(array(int))):   Return multiple generated team drafts based on different sampling strategies
    '''

    # Top 10 champions based on probability
    # Champion probability up to 50%
    # Low temperature to view highest probability champs for each role
    sampling_configs = [
        ("top_k", {"k": 10}, top_k_drafts),
        ("top_p", {"p": 0.2}, top_p_drafts),
        ("temperature", {"temperature": 1e-8}, temp_drafts)
    ]
    drafts = {
        "top_k": [],
        "top_p": [],
        "temperature": []
    }

    for method, params, num_drafts in sampling_configs:
        for i in range(num_drafts):
            generated_draft = generate_team(
                model,
                riot_champ_ids,
                device=device,
                sampling_method=method,
                **params
            )
            draft = [Id_to_Champion[Consecutive_Id_to_Id[champ]] for champ in generated_draft]
            drafts[method].append(draft[5:])

    return drafts

# Create role-based champion lists for selection
champion_roles = {
    'Top': sorted(list(Champion_to_Id.keys())),
    'Jungle': sorted(list(Champion_to_Id.keys())),
    'Mid': sorted(list(Champion_to_Id.keys())),
    'ADC': sorted(list(Champion_to_Id.keys())),
    'Support': sorted(list(Champion_to_Id.keys()))
}

# Initialize device and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {
    'KR': RoleAwareTransformer(),
    'NA': RoleAwareTransformer(),
    'EUW': RoleAwareTransformer()
}

# Initialize outcome predictor models
outcome_predictors = {
    'KR': LogisticRegressionModel(input_dim=(len(Champion_to_Id)*10)),
    'NA': LogisticRegressionModel(input_dim=(len(Champion_to_Id)*10)),
    'EUW': LogisticRegressionModel(input_dim=(len(Champion_to_Id)*10))
}

# Load model weights
models['KR'].load_state_dict(torch.load('models/kr_challenger_model_2_transformer.pt', map_location=device))
models['NA'].load_state_dict(torch.load('models/na1_challenger_model_2_transformer.pt', map_location=device))
models['EUW'].load_state_dict(torch.load('models/euw_challenger_model_2_transformer.pt', map_location=device))

# Load outcome predictor weights
outcome_predictors['KR'].load_state_dict(torch.load('models/kr_challenger_model_1_binary_predictor.pt', map_location=device))
outcome_predictors['NA'].load_state_dict(torch.load('models/na1_challenger_model_1_binary_predictor.pt', map_location=device))
outcome_predictors['EUW'].load_state_dict(torch.load('models/euw_challenger_model_1_binary_predictor.pt', map_location=device))

# Set all models to evaluation mode
for model in models.values():
    model.to(device)
    model.eval()

for predictor in outcome_predictors.values():
    predictor.to(device)
    predictor.eval()

@app.route('/')
def home():
    all_champions = sorted(list(Champion_to_Id.keys()))
    return render_template('index.html', champion_roles=champion_roles, all_champions=all_champions)

@app.route('/generate_draft', methods=['POST'])
def generate_draft():
    data = request.get_json()
    team = data.get('team', {})
    region = data.get('region', 'KR')
    
    # Convert champion names to IDs
    champion_ids = []  
    for role in ['top', 'jungle', 'mid', 'adc', 'support']:
        champ_name = team.get(role, '')
        if champ_name in Champion_to_Id:
            champion_ids.append(Champion_to_Id[champ_name])
        else:
            return jsonify({'error': f'Invalid champion name for {role}: {champ_name}'}), 400
    
    # Generate draft using the selected region's model
    model = models[region]
    outcome_predictor = outcome_predictors[region]
    try:
        drafts = generate_multiple_drafts(
            model,
            champion_ids,
            device=device,
            top_k_drafts=5,
            top_p_drafts=5,
            temp_drafts=1
        )
        
        if drafts is None:
            return jsonify({'error': 'Failed to generate drafts'}), 500
            
        # Convert the draft data to a serializable format
        highest_probability = 0
        highest_probability_draft = None
        
        for method, draft_list in drafts.items():
            for draft in draft_list:
                # Convert champion names back to IDs for prediction
                enemy_ids = champion_ids
                ally_ids = [Champion_to_Id[champ] for champ in draft]
                all_ids = enemy_ids + ally_ids
                
                # Get win probability using the outcome predictor
                win_probability = float(outcome_predictor.predict_outcome(all_ids, device=device))
                if win_probability > highest_probability:
                    highest_probability = win_probability
                    highest_probability_draft = {
                        'generated_team': draft,
                        'win_probability': win_probability
                    }
        
        return jsonify({'drafts': [highest_probability_draft]})
    except Exception as e:
        print(f"Error generating draft: {str(e)}")
        return jsonify({'error': 'Failed to generate draft'}), 500

if __name__ == '__main__':
    app.run(debug=True)