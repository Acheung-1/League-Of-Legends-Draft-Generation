import random
from torch.utils.data import DataLoader, TensorDataset, random_split
import ast

def read_txt_match_data(file_path):
    '''
    DESCRIPTION:
        Reads a txt file containing match data (each line is a list)
        Removes the first element (patch) from each list
        Returns a list of the containing champion ids (team 1 [losing] and team 2 [winning])

    INPUTS:
        file_path (str):        File path to txt file

    OUTPUTS:
        list[list[int]]:        A list of lists containing champion IDs, patch is removed
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            array = []
            for line in f:
                line = ast.literal_eval(line)
                array.append(line[1:])
            return array
    except FileNotFoundError:
        return
    
def append_to_txt(item, file_path):
    '''
    DESCRIPTION:
        Appends a string to a txt file 
    
    INPUTS:
        Item (str):             Item to append to txt file
        file_path (str):        File path to txt file
    '''
    with open(file_path, 'a') as f:
        f.write(str(item) + '\n')
    return
    
def main():
    '''
    DESCRIPTION:
        Reads a txt file that is currently formated [losing team, winning team]
        Chooses a random number between 0 and 1
        Randomizes order of losing team, winning team
            If < 0.5, format  x = [winning team, losing team] y = [0]
            If >= 0.5, format x = [losing team, winning team] y = [1]
        Stores x and y in distinct txt file to use as training data
    
    
    OUTPUTS:
        x_data.txt (txt file):  [team 1, team 2]        
        y_data.txt (txt file):  outcome of team 2 --- losing = 0, winning = 1
    '''

    match_data = read_txt_match_data("training_data.txt")
    x = []
    y = []
    for line in match_data:
        random_num = random.random()
        # Store the outcome of being the second team
        if random_num < 0.5: # [winning team, losing team] [0]
            reverse = line[5:]+line[:5]
            x.append(reverse)
            y.append(0)
        else: # [losing team, winning team] [1]
            x.append(line)
            y.append(1)
    for index in range(len(x)):
        append_to_txt(x[index], "x_data.txt")
        append_to_txt(y[index], "y_data.txt")



main()
