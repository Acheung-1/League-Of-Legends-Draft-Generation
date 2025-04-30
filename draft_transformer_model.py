import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
import ast
import random
from champion_dictionary import Champion_to_Id, Id_to_Champion, Id_to_Consecutive_Id, Consecutive_Id_to_Id

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
                    team_comps = ast.literal_eval(line)
                    team_comps = [Id_to_Consecutive_Id[int(x)] for x in team_comps]
                    if len(team_comps) != 10:
                        continue
                    enemy = team_comps[:5]
                    predicted_team = team_comps[5:]
                    if any(champ < 0 or champ >= self.vocab_size for champ in team_comps):
                        continue
                    input_vec = enemy + [0]*5
                    target_vec = predicted_team
                    self.data.append((input_vec, target_vec))
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
    def __init__(self, vocab_size, embed_dim=128, use_role_embedding=True):
        super().__init__()
        self.use_role_embedding = use_role_embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if self.use_role_embedding:
            self.role_embedding = nn.Embedding(10, embed_dim)  # 10 positions: 5 enemy, 5 win team

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
            num_layers=2
        )

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: [batch, 10]
        emb = self.embedding(x)  # [batch, 10, embed_dim]

        if self.use_role_embedding:
            role_ids = torch.arange(10, device=x.device).unsqueeze(0).repeat(x.size(0), 1)
            role_emb = self.role_embedding(role_ids)
            emb = emb + role_emb  # Combine champ + role info

        out = self.transformer(emb)  # [batch, 10, embed_dim]
        win_out = out[:, 5:, :]  # Keep last 5: predicted team
        logits = self.fc(win_out)  # [batch, 5, vocab_size]
        return logits


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
            logits = model(x)  # [batch, 5, vocab_size]
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))  # flatten for loss
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

def generate_team(model, input_seq, vocab_size, device, sampling_method="top_k", **sampling_params):
    model.eval()
    team = input_seq[:5]  # Enemy team
    generated = []

    for i in range(5):
        current_input = team + generated + [0] * (5 - len(generated))
        x = torch.tensor(current_input, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[0]  # Shape: (5, vocab_size)
        logits_i = logits[len(generated)]  # logits for current position
        next_champ = generate_with_randomness(logits_i, method=sampling_method, **sampling_params)

        # Avoid duplicates
        while next_champ in team + generated:
            next_champ = generate_with_randomness(logits_i, method=sampling_method, **sampling_params)

        generated.append(next_champ)

    return team + generated


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        count = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            predictions = torch.argmax(logits, dim=2)  # (batch_size, 5)

            for i in range(5):  # For positions 0–4 (winning team slots)
                correct += (predictions[:, i] == y[:, i]).sum().item()
            total += x.size(0) * 5  # 5 champs per sample

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

def decode_champ_list(champ_ids):
    '''
    DESCRIPTION:
        Takes an array of champions that 

    INPUTS:
        champ_ids (array(int)):    Array containing champions in consecutive ids for training

    OUTPUTS:
        Output (type):             Array containing champion names
    '''
    return [Id_to_Champion[Consecutive_Id_to_Id[champ_id]] for champ_id in champ_ids]

def generate_multiple_drafts(model, seed, vocab_size, device, top_k_drafts=3, top_p_drafts=3, temp_drafts=1):
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
        ("top_k", {"k": 3}, top_k_drafts),
        ("top_p", {"p": 0.5}, top_p_drafts),
        ("temperature", {"temperature": 1e-8}, temp_drafts)
    ]

    # Prepare the seed
    print(f"Losing Team + Winning Team: {decode_champ_list(seed)}\n")
    print(f"Enemy Team: {decode_champ_list(seed[:5])}\n")
    seed = seed[:5] + [0,0,0,0,0]  # Keep first 5 champions and pad with zeros

    for method, params, num_drafts in sampling_configs:
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
            draft = [Id_to_Champion[Consecutive_Id_to_Id[champ]] for champ in generated_draft]
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

    # Initialize and train model
    model = RoleAwareTransformer(vocab_size=vocab_size, embed_dim=embed_dim)
    train_model(model, train_loader, vocab_size=vocab_size, epochs=10, device=device)
    torch.save(model.state_dict(), 'draft_predictor_model_na_challenger.pt')
    print("Training complete. Model saved to na_challenger_model.pt.")

    # # evaluate_model(model, test_loader, device)

    # Initialize model with same architecture
    model = RoleAwareTransformer(vocab_size=vocab_size, embed_dim=128)

    # Add RoleAwareTransformer to safe globals list
    # torch._utils.add_safe_globals([RoleAwareTransformer])
    
    # Load the saved model weights
    
    #How saved it
    # model = torch.load('draft_predictor_model_na_challenger.pt', map_location=device, weights_only=False)
    # model.eval()
    model = RoleAwareTransformer(vocab_size=vocab_size, embed_dim=embed_dim)  # Create the model first
    model.load_state_dict(torch.load('draft_predictor_model_na_challenger.pt', map_location=device))  # Load weights
    model.to(device)
    model.eval()

    # Example usage
    seed = [150, 80, 268, 110, 235, 897, 234, 4, 81, 117]  # patch + losing team (5 picks total)
    # seed = [777, 254, 711, 901, 40, 420, 245, 142, 221, 53]  # patch + losing team (5 picks total)
    # seed = [0, 0, 0, 901, 0, 420, 245, 142, 221, 53]  # patch + losing team (5 picks total)
    # draft = [Id_to_Champion[champ] for champ in seed]
    generate_multiple_drafts(model,seed,vocab_size=vocab_size,device=device,top_k_drafts=3, top_p_drafts=3, temp_drafts=1)


if __name__ == "__main__":
    main()