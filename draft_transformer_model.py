import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
import ast
from champion_dictionary import Champion_to_Id, Id_to_Champion, Id_to_Consecutive_Id, Consecutive_Id_to_Id

# ---------- Dataset ----------
class DraftDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.vocab_size = get_vocab_size()
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

def get_vocab_size():
    return len(Id_to_Champion)

def decode_champ_list(champ_ids):
    '''
    DESCRIPTION:
        Takes an array of CONSECUTIVE champion IDS, and return champion name

    INPUTS:
        champ_ids (array(int)):    Array containing champions in consecutive ids for training

    OUTPUTS:
        Output (type):             Array containing champion names
    '''
    return [Id_to_Champion[champ_id] for champ_id in champ_ids]

# ---------- Model ----------
class RoleAwareTransformer(nn.Module):
    def __init__(self, vocab_size=get_vocab_size(), embed_dim=128, use_role_embedding=True):
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
def train_model(model, dataloader, vocab_size, epochs=20, device='cpu'):
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

def evaluate_model(model, dataloader, device, sampling_method="temperature", sampling_params={"temperature": 1e-8}):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            for i in range(batch_size):
                input_seq = x[i].tolist()
                target_seq = y[i].tolist()

                generated_seq = generate_team(
                    model, input_seq, device,
                    sampling_method=sampling_method,
                    **sampling_params
                )
                predicted = generated_seq[5:]  # Get the generated winning team
                correct += sum([predicted[j] == target_seq[j] for j in range(5)])
                total += 5  # 5 positions per sample

    accuracy = correct / total if total > 0 else 0
    print(f"Partial Test Accuracy using {sampling_method} (params={sampling_params}): {accuracy * 100:.2f}%")

# ---------- Sampling Strategies ----------
def top_k_sampling(logits, k=10):
    '''
    DESCRIPTION:
        Performs top-k sampling by selecting from the k most probable logits, then sampling one
    
    INPUTS:
        logits (torch.Tensor):     Logits vector (1D tensor of size vocab_size)
        k (int):                   Number of top logits to consider during sampling
    
    OUTPUTS:
        sampled_index (int):       Index of the sampled champion
    '''
    topk_probs, topk_indices = torch.topk(logits, k)
    topk_probs = torch.softmax(topk_probs, dim=-1)
    sampled_idx = torch.multinomial(topk_probs, 1)
    return topk_indices[sampled_idx].item()

def top_p_sampling(logits, p):
    '''
    DESCRIPTION:
        Performs nucleus (top-p) sampling from logits. Chooses smallest set of logits whose cumulative probability exceeds p,
        then samples from this set
    
    INPUTS:
        logits (torch.Tensor):     Logits vector (1D tensor of size vocab_size)
        p (float):                 Cumulative probability threshold (0 < p <= 1)
    
    OUTPUTS:
        sampled_index (int):       Index of the sampled champion
    '''
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
    '''
    DESCRIPTION:
        Applies temperature scaling to logits and samples from the resulting probability distribution
    
    INPUTS:
        logits (torch.Tensor):     Logits vector (1D tensor of size vocab_size)
        temperature (float):       Value to scale logits (higher = more random, lower = more confident)
    
    OUTPUTS:
        sampled_index (int):       Index of the sampled champion
    '''
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def generate_with_randomness(logits, method="top_k", **kwargs):
    '''
    DESCRIPTION:
        Dispatches logits to the appropriate sampling method: top-k, top-p, temperature, or greedy
    
    INPUTS:
        logits (torch.Tensor):     Logits vector (1D tensor of size vocab_size)
        method (str):              Sampling method to use: "top_k", "top_p", "temperature", or "argmax"
        **kwargs:                  Sampling-specific arguments
    
    OUTPUTS:
        sampled_index (int):       Index of the sampled champion
    '''

    if method == "top_k":
        return top_k_sampling(logits, kwargs.get("k", 10))
    elif method == "top_p":
        return top_p_sampling(logits, kwargs.get("p", 0.9))
    elif method == "temperature":
        return sample_with_temperature(logits, kwargs.get("temperature", 1.0))
    else:
        return torch.argmax(logits).item()

def generate_team(model, input_seq, device, max_retries=1, sampling_method="top_k", **sampling_params):
    '''
    DESCRIPTION:
        Autoregressively generates a winning team draft given an enemy team and sampling method
    
    INPUTS:
        model (nn.Module):         Trained draft prediction model
        input_seq (list[int]):     List of 10 ints — 5 enemy champions, 5 zeros
        device (str):              'cuda' or 'cpu'
        max_retries (int):           Max attempts to avoid duplicates per position
        sampling_method (str):     Method used for sampling ("top_k", "top_p", "temperature", "argmax")
        **sampling_params:         Additional sampling parameters like k, p, or temperature
    
    OUTPUTS:
        full_draft (list[int]):    List of 10 ints — 5 enemy, 5 generated winning team champions
    '''
    model.eval()
    team = [Id_to_Consecutive_Id[id] for id in input_seq]
    team = team[:5]
    generated = []

    for i in range(5):
        current_input = team + generated + [0] * (5 - len(generated))
        x = torch.tensor(current_input, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[0]  # Shape: (5, vocab_size)
        logits_i = logits[len(generated)]  # logits for current position

        attempts = 0
        next_champ = None
        used = set(team + generated)

        while attempts < max_retries:
            candidate = generate_with_randomness(logits_i, method=sampling_method, **sampling_params)
            if candidate not in used:
                next_champ = candidate
                break
            attempts += 1

        # Fallback to highest available champ if needed
        if next_champ is None:
            sorted_logits = torch.argsort(logits_i, descending=True)
            for champ in sorted_logits:
                champ_id = champ.item()
                if champ_id not in used:
                    next_champ = champ_id
                    break

        if next_champ is None:
            raise RuntimeError("No valid champion found for generation, even after fallback.")

        generated.append(next_champ)

    return team + generated

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
        None (prints results to console)
    '''
    sampling_configs = [
        ("top_k", {"k": 3}, top_k_drafts),
        ("top_p", {"p": 0.5}, top_p_drafts),
        ("temperature", {"temperature": 1e-8}, temp_drafts)
    ]

    # Print out sequence
    print(f"Enemy Team: {decode_champ_list(riot_champ_ids[:5])}\n")
    print(f"Winning Team: {decode_champ_list(riot_champ_ids[5:])}\n")

    for method, params, num_drafts in sampling_configs:
        print(f"\nGenerating {num_drafts} drafts using {method} sampling:")
        for i in range(num_drafts):
            generated_draft = generate_team(
                model,
                riot_champ_ids,
                device=device,
                sampling_method=method,
                **params
            )
            draft = [Id_to_Champion[Consecutive_Id_to_Id[champ]] for champ in generated_draft]
            print(f"Draft {i+1}: {draft[5:]}")
        print("-" * 50)

def main(formatted_training_data="formatted_training_data.txt",
         transformer_draft_generator_model="transformer_model.pt"
):
    '''
    DESCRIPTION:
        Loads the dataset, trains a RoleAwareTransformer model on champion draft data
        evaluates its performance, and saves the trained model
    
    INPUTS:
        formatted_training_data (str):              Path to training data file (each line is a 10-element list of champion IDs)
        transformer_draft_generator_model (str):    Path to save the trained PyTorch model
    
    OUTPUTS:
        transformer_draft_generator_model (str):    Path to save the trained PyTorch model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is NOT available. Using CPU.")

    vocab_size = get_vocab_size()

    batch_size = 32
    embed_dim = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset and split
    full_dataset = DraftDataset(formatted_training_data)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize and train model
    model = RoleAwareTransformer(vocab_size=vocab_size, embed_dim=embed_dim)
    train_model(model, train_loader, vocab_size=vocab_size, epochs=20, device=device)
    torch.save(model.state_dict(), transformer_draft_generator_model)
    print(f"Training complete. Model saved to {transformer_draft_generator_model}")

    # evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()