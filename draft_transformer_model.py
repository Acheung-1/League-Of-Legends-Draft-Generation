import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
import ast
import random
from champion_dictionary import Champion_to_Id, Id_to_Champion

# ---------- Dataset ---------- 
# predicting each champion one by one
class DraftDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.vocab_size = get_vocab_size(file_path)
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    full = ast.literal_eval(line)
                    full = [int(float(x)) for x in full]  # 🔥 Convert everything to int
                    if len(full) < 10:
                        continue
                    for i in range(5, 10):  # Predict each winning champ position
                        target = full[i]
                        if target >= self.vocab_size or target < 0:
                            continue  # skip bad values
                        input_vec = full[:i] + [0] * (10 - i)
                        self.data.append((input_vec, target))
                except Exception as e:
                    print(f"Skipping line due to error: {e}")
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ---------- Model ----------
class RoleAwareTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super(RoleAwareTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
            num_layers=2 # Increase this to play around
        )
        self.fc = nn.Linear(embed_dim * 11, vocab_size)  # Changed to 11 tokens

    def forward(self, x):
        emb = self.embedding(x)  
        out = self.transformer(emb)  
        out = out.flatten(start_dim=1)  
        return self.fc(out) 

# ---------- Training ----------
def train_model(model, dataloader, vocab_size, epochs=10, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# ---------- Sampling Strategies ----------
def top_k_sampling(logits, k=10):
    topk_probs, topk_indices = torch.topk(logits, k)
    topk_probs = torch.softmax(topk_probs, dim=-1)
    sampled_idx = torch.multinomial(topk_probs, 1)
    return topk_indices[sampled_idx].item()

def top_p_sampling(logits, p):
    """Performs nucleus (top-p) sampling from logits."""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative_probs > p

    if not torch.any(cutoff):
        cutoff_idx = len(sorted_probs)
    else:
        cutoff_idx = (cutoff.nonzero(as_tuple=True)[0][0] + 1).item()

    filtered_probs = sorted_probs[:cutoff_idx]
    filtered_indices = sorted_indices[:cutoff_idx]

    filtered_probs /= filtered_probs.sum()  # renormalize to sum to 1

    chosen_index = torch.multinomial(filtered_probs, 1).item()
    return filtered_indices[chosen_index].item()


def sample_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def generate_with_randomness(logits, method="top_k", **kwargs):
    if method == "top_k":
        return top_k_sampling(logits, kwargs.get("k", 10))
    elif method == "top_p":
        return top_p_sampling(logits, kwargs.get("p", 0.9))
    elif method == "temperature":
        return sample_with_temperature(logits, kwargs.get("temperature", 1.0))
    else:
        return torch.argmax(logits).item()

# ---------- Generation Function ----------
def generate_team(model, input_seq, vocab_size, device, sampling_method="top_k", **sampling_params):
    model.eval()
    input_seq = input_seq[:5]  # Keep losing team 
    team = input_seq[:5]  # Start with losing team

    for _ in range(5, 10):  # Predict winning team
        current_input = team + [0] * (10 - len(team))  # Pad to length 10
        x = torch.tensor(current_input, dtype=torch.long, device=device).unsqueeze(0)  # (1, 10)
        logits = model(x)[0]

        # Sample the next champion for the current position
        next_champ = generate_with_randomness(logits, method=sampling_method, **sampling_params)

        # Avoid picking the same champion twice in the same team
        while next_champ in team:
            next_champ = generate_with_randomness(logits, method=sampling_method, **sampling_params)

        team.append(next_champ)
    return team

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        count = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            predictions = torch.argmax(logits, dim=1)  # Predicted champions for positions 6–10

            # Count how many champions are correctly predicted
            for i in range(5, 10):  # Check for positions 6 to 10
                if predictions[i] == y[i]:  # If predicted champion matches the true champion
                    correct += 1
            total += 5  # We're checking 5 positions in total (positions 6 to 10)

    accuracy = correct / total if total > 0 else 0
    print(f"Partial Test Accuracy: {accuracy * 100:.2f}%")

def get_vocab_size(file_path):
    max_token = 0
    with open(file_path, "r") as f:
        for line in f:
            try:
                tokens = ast.literal_eval(line.strip())
                if tokens:
                    max_token = max(max_token, max(tokens))
            except:
                continue
    return max_token + 1  # +1 since torch assumes class labels from 0 to vocab_size-1

def generate_multiple_drafts(model, seed, vocab_size, device, num_drafts=5):
    '''
    DESCRIPTION:
        Generate multiple drafts using different sampling methods
    
    INPUTS:
        model ():                  The trained model
        seed (array(int)):         Losing and winning team draft
        vocab_size (int):          Size of the champion vocabulary
        device:                    Computing device (cuda/cpu)
        num_drafts (int):          Number of drafts to generate for each sampling method
    
    OUTPUTS:
        Output (type):             description
    '''
    sampling_configs = [
        ("top_k", {"k": 3}),
        ("top_p", {"p": 0.1}),
        ("temperature", {"temperature": 1e-8})
    ]
    
    # Prepare the seed
    print(f"Losing Team + Winning Team: {[Id_to_Champion[champ] for champ in seed]}\n")
    print(f"Enemy Team: {[Id_to_Champion[champ] for champ in seed[:5]]}\n")
    seed = seed[:5] + [0,0,0,0,0]  # Keep first 5 champions and pad with zeros
    
    for method, params in sampling_configs:
        print(f"\nGenerating {num_drafts} drafts using {method} sampling:")
        for i in range(num_drafts):
            generated_draft = generate_team(
                model,
                seed,
                vocab_size=vocab_size,
                device=device,
                sampling_method=method,
                **params
            )
            draft = [Id_to_Champion[champ] for champ in generated_draft]
            print(f"Draft {i+1}: {draft[5:]}")
        print("-" * 50)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is NOT available. Using CPU.")

    file_path = "formatted_training_data.txt"
    vocab_size = get_vocab_size(file_path)
    print("Vocab size:", vocab_size)

    batch_size = 32
    embed_dim = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset and split
    full_dataset = DraftDataset(file_path)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # # Initialize and train model
    # model = RoleAwareTransformer(vocab_size=vocab_size, embed_dim=embed_dim)
    # train_model(model, train_loader, vocab_size=vocab_size, epochs=10, device=device)
    # torch.save(model.state_dict(), 'model.pt')
    # print("Training complete. Model saved to na_challenger_model.pt.")

    # # evaluate_model(model, test_loader, device)

    # Initialize model with same architecture
    model = RoleAwareTransformer(vocab_size=vocab_size, embed_dim=128)

    # Add RoleAwareTransformer to safe globals list
    # torch._utils.add_safe_globals([RoleAwareTransformer])
    
    # Load the saved model weights
    
    #How saved it
    model = torch.load('draft_predictor_model_na_challenger.pt', map_location=device, weights_only=False)
    # model.eval()

    # # should be saved like: torch.save(model.state_dict(), 'model.pt')
    # model.load_state_dict(torch.load('model.pt', map_location=device))
    # model.to(device)
    # # model.eval()

    # Example usage
    seed = [150, 80, 268, 110, 235, 897, 234, 4, 81, 117]  # patch + losing team (5 picks total)
    # seed = [777, 254, 711, 901, 40, 420, 245, 142, 221, 53]  # patch + losing team (5 picks total)
    # seed = [0, 0, 0, 901, 0, 420, 245, 142, 221, 53]  # patch + losing team (5 picks total)
    # draft = [Id_to_Champion[champ] for champ in seed]
    generate_multiple_drafts(model,seed,vocab_size=vocab_size,device=device,num_drafts=3)


    

main()