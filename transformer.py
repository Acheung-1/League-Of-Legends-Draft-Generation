import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
import ast
import random

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is NOT available. Using CPU.")

# ---------- Dataset ---------- 
# predicting each champion one by one
class DraftDataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    full = ast.literal_eval(line)
                    if len(full) < 11:
                        continue
                    for i in range(6, 11):
                        target = full[i]
                        if target == 0 or target >= 2000 or target < 0:
                            continue  
                        input_vec = full[:i] + [0] * (11 - i)
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

def top_p_sampling(logits, p=0.9):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative_probs > p
    cutoff_idx = torch.argmax(cutoff).item() + 1
    filtered_probs = sorted_probs[:cutoff_idx]
    filtered_indices = sorted_indices[:cutoff_idx]
    sampled_idx = torch.multinomial(filtered_probs, 1)
    return filtered_indices[sampled_idx].item()

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
    input_seq = input_seq[:6]  # Keep patch and losing team 
    team = input_seq[:6]  # Start with patch and losing team

    for _ in range(6, 11):  # Predict winning team
        current_input = team + [0] * (11 - len(team))  # Pad to length 11
        x = torch.tensor(current_input, dtype=torch.long, device=device).unsqueeze(0)  # (1, 11)
        logits = model(x)[0]

        # Sample the next champion for the current position
        next_champ = generate_with_randomness(logits, method=sampling_method, **sampling_params)

        # Avoid picking the same champion twice in the same team
        while next_champ in team:
            next_champ = generate_with_randomness(logits, method=sampling_method, **sampling_params)

        team.append(next_champ)
    return team



main()