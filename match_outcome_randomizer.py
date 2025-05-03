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

def save_lists_to_file(lists, file_path):
    '''
    DESCRIPTION:
        Save a list of lists to a text file, with each list on a new line
    
    INPUTS:
        lists (array(array(ints))):     List of lists to save
        file_path (str):                File path to txt file
    '''
    with open(file_path, 'w') as f:
        for item in lists:
            f.write(str(item) + '\n')
    
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
    
def main(unformatted_training_data="training_data.txt", 
         formatted_training_data="formatted_training_data.txt",
         randomized_team_comp="x_randomized_team_comp.txt",
         randomized_outcomes="y_randomized_outcomes.txt",
):
    '''
    DESCRIPTION:
        Reads a txt file that is currently formated [patch, losing team, winning team]
        Removes patch from the each list
        Chooses a random number between 0 and 1
        Randomizes order of losing team, winning team
            If < 0.5, format  x = [winning team, losing team] y = [0]
            If >= 0.5, format x = [losing team, winning team] y = [1]
        Stores x and y in distinct txt file to use as training data
    
    
    OUTPUTS:
        x_data.txt (txt file):  [team 1, team 2]        
        y_data.txt (txt file):  outcome of team 2 --- losing = 0, winning = 1
    '''

    match_data = read_txt_match_data(unformatted_training_data)
    save_lists_to_file(match_data, formatted_training_data)
    match_team_comp = []
    outcome = []
    for line in match_data:
        random_num = random.random()
        # Store the outcome of being the second team
        if random_num < 0.5: # [winning team, losing team] [0]
            reverse = line[5:]+line[:5]
            match_team_comp.append(reverse)
            outcome.append(0)
        else: # [losing team, winning team] [1]
            match_team_comp.append(line)
            outcome.append(1)
    for index in range(len(match_team_comp)):
        append_to_txt(match_team_comp[index], randomized_team_comp)
        append_to_txt(outcome[index], randomized_outcomes)

if __name__ == "__main__":
    main()