import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import ast

def read_txt_to_x_tensor(file_path,dtype=torch.int64):
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
            return torch.tensor(data,dtype=dtype)
    except FileNotFoundError:
        return None
    
def read_txt_to_y_tensor(file_path,dtype=torch.int64):
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
            return torch.tensor(data,dtype)
    except FileNotFoundError:
        return None

def ids_to_one_hot(ids):
    num_champions = len(Id_to_Champion)
    one_hot_matrix = []
    for champ_id in ids:
        array = [0]*num_champions
        array[Id_to_Consecutive_Id[champ_id]] = 1
        one_hot_matrix += array
    return one_hot_matrix