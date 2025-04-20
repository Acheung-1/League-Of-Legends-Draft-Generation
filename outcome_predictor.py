import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import ast
from champion_dictionary import Id_to_Champion, Champion_to_Id, Id_to_Consecutive_Id


def read_txt_to_x_tensor(file_path, dtype=torch.float32):
    '''
    DESCRIPTION:
        Reads a .txt file where each line is a list of ints (e.g., match data).
        Converts to a PyTorch tensor.

    INPUTS:
        file_path (str):            Path to the .txt file.
        dtype (torch.dtype):        Desired tensor dtype (default: torch.int64)

    OUTPUTS:
        torch.Tensor:               Tensor of shape (n_samples, n_features)
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [ids_to_one_hot(ast.literal_eval(line)) for line in f]
            return torch.tensor(data, dtype=dtype)
    except FileNotFoundError:
        return None
    
def read_txt_to_y_tensor(file_path, dtype=torch.float32):
    '''
    DESCRIPTION
        Reads a .txt file where each line an ints (outcome data - 0 or 1)
        Converts to a PyTorch tensor

    INPUTS:
        file_path (str):            Path to the .txt file
        dtype (torch.dtype):        Desired tensor dtype (default: torch.int64)

    OUTPUTS:
        (torch.Tensor):             Tensor of shape (n_samples, n_features)
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [ast.literal_eval(line) for line in f]
            return torch.tensor(data, dtype=dtype)
    except FileNotFoundError:
        return None

def ids_to_one_hot(ids):
    '''
    DESCRIPTION:
        Converts an array of champion IDs into a flattened one-hot encoded vector
        Each champion is represented as a one-hot vector of length equal to the number of champions
        Vectors are concatenated into a single long vector
    
    INPUTS:
        ids (array(ints)):         array of champion ids used by riot api
    
    OUTPUTS:
        one_hot_matrix (type):     A flattened vector containing the one-hot encoding of each ID
    '''
    num_champions = len(Id_to_Champion)
    one_hot_matrix = []
    for champ_id in ids:
        array = [0]*num_champions
        array[Id_to_Consecutive_Id[champ_id]] = 1
        one_hot_matrix += array
    return one_hot_matrix

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    

def main():
    '''
    DESCRIPTION:
        Trains a logistic regression model on input features and binary labels using PyTorch
            1. Loads data from text files into tensors
            2. Splits data into training and test sets
            3. Trains a logistic regression model using the binary cross-entropy loss
            4. Evaluates model accuracy on the test set
            5. Saves the trained model to disk

    OUTPUTS:
        outcome_predictor_model.pt (torch model):       The trained model is saved to this file
    '''
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print what you're using
    if device.type == "cuda":
        print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is NOT available. Using CPU.")
    
    X = read_txt_to_x_tensor('x_data.txt')           # shape: (n_samples, 10)
    y = read_txt_to_y_tensor('y_data.txt').reshape(-1, 1)  # shape: (n_samples, 1)

    # Create dataset
    dataset = TensorDataset(X, y)

    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Instantiate model
    input_dim = X.shape[1]
    model = LogisticRegressionModel(input_dim).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X=batch_X.to(device)
            batch_y=batch_y.to(device)
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Evaluate model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predicted = (outputs >= 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = correct / total
        print(f'Accuracy on test set: {accuracy:.4f}')
    
    torch.save(model,"outcome_predictor_model_na_challenge.pt")
    print("Training complete. Model saved to outcome_predictor_model_na_challenge.pt")  

main()