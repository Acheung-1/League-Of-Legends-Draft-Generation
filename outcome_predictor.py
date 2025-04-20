import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import ast

Id_to_Champion = {
    0:"None",
    1:"Annie",
    10:"Kayle",
    101:"Xerath",
    102:"Shyvana",
    103:"Ahri",
    104:"Graves",
    105:"Fizz",
    106:"Volibear",
    107:"Rengar",
    11:"Master Yi",
    110:"Varus",
    111:"Nautilus",
    112:"Viktor",
    113:"Sejuani",
    114:"Fiora",
    115:"Ziggs",
    117:"Lulu",
    119:"Draven",
    12:"Alistar",
    120:"Hecarim",
    121:"Kha'Zix",
    122:"Darius",
    126:"Jayce",
    127:"Lissandra",
    13:"Ryze",
    131:"Diana",
    133:"Quinn",
    134:"Syndra",
    136:"Aurelion Sol",
    14:"Sion",
    141:"Kayn",
    142:"Zoe",
    143:"Zyra",
    145:"Kai'Sa",
    147:"Seraphine",
    15:"Sivir",
    150:"Gnar",
    154:"Zac",
    157:"Yasuo",
    16:"Soraka",
    161:"Vel'koz",
    163:"Taliyah",
    164:"Camille",
    166:"Akshan",
    17:"Teemo",
    18:"Tristana",
    19:"Warwick",
    2:"Olaf",
    20:"Nunu",
    200:"Bel'Veth",
    201:"Braum",
    202:"Jhin",
    203:"Kindred",
    21:"Miss Fortune",
    22:"Ashe",
    221:"Zeri",
    222:"Jinx",
    223:"Tahm Kench",
    23:"Tryndamere",
    233:"Briar",
    234:"Viego",
    235:"Senna",
    236:"Lucian",
    238:"Zed",
    24:"Jax",
    240:"Kled",
    245:"Ekko",
    246:"Qiyana",
    25:"Morgana",
    254:"Vi",
    26:"Zilean",
    266:"Aatrox",
    267:"Nami",
    268:"Azir",
    27:"Singed",
    28:"Evelyn",
    29:"Twitch",
    3:"Galio",
    30:"Karthus",
    31:"Cho'Gath",
    32:"Amumu",
    33:"Rammus",
    34:"Anivia",
    35:"Shaco",
    350:"Yuumi",
    36:"Dr. Mundo",
    360:"Samira",
    37:"Sona",
    38:"Kassadin",
    39:"Irelia",
    4:"Twisted Fate",
    40:"Janna",
    41:"Gangplank",
    412:"Thresh",
    42:"Corki",
    420:"Illaoi",
    421:"Rek'Sai",
    427:"Ivern",
    429:"Kalista",
    43:"Karma",
    432:"Bard",
    44:"Taric",
    45:"Veigar",
    48:"Trundle",
    497:"Rakan",
    498:"Xayah",
    5:"Xin Zhao",
    50:"Swain",
    51:"Caitlyn",
    516:"Ornn",
    517:"Sylas",
    518:"Neeko",
    523:"Aphelious",
    526:"Rell",
    53:"Blitzcrank",
    54:"Malphite",
    55:"Katarina",
    555:"Pyke",
    56:"Nocturne",
    57:"Maokai",
    58:"Renekton",
    59:"Jarvan IV",
    6:"Urgot",
    60:"Elise",
    61:"Orianna",
    62:"Wukong",
    63:"Brand",
    64:"Lee Sin",
    67:"Vayne",
    68:"Rumble",
    69:"Cassiopia",
    7:"Leblanc",
    711:"Vex",
    72:"Skarner",
    74:"Heimerdinger",
    75:"Nasus",
    76:"Nidalee",
    77:"Udyr",
    777:"Yone",
    78:"Poppy",
    79:"Gragas",
    799:"Ambessa",
    8:"Vladamir",
    80:"Pantheon",
    800:"Mel",
    81:"Ezreal",
    82:"Mordekaiser",
    83:"Yorick",
    84:"Akali",
    85:"Kennen",
    86:"Garen",
    875:"Sett",
    876:"Lilia",
    887:"Gwen",
    888:"Renata Glasc",
    89:"Leona",
    893:"Aurora",
    895:"Nilah",
    897:"K'Sante",
    9:"Fiddlesticks",
    90:"Malzahar",
    901:"Smolder",
    902:"Milio",
    91:"Talon",
    910:"Hwei",
    92:"Riven",
    950:"Naafiri",
    96:"Kog'Maw",
    98:"Shen",
    99:"Lux",
}

Id_to_Consecutive_Id = {
    0:0,
    1:1,
    10:2,
    101:3,
    102:4,
    103:5,
    104:6,
    105:7,
    106:8,
    107:9,
    11:10,
    110:11,
    111:12,
    112:13,
    113:14,
    114:15,
    115:16,
    117:17,
    119:18,
    12:19,
    120:20,
    121:21,
    122:22,
    126:23,
    127:24,
    13:25,
    131:26,
    133:27,
    134:28,
    136:29,
    14:30,
    141:31,
    142:32,
    143:33,
    145:34,
    147:35,
    15:36,
    150:37,
    154:38,
    157:39,
    16:40,
    161:41,
    163:42,
    164:43,
    166:44,
    17:45,
    18:46,
    19:47,
    2:48,
    20:49,
    200:50,
    201:51,
    202:52,
    203:53,
    21:54,
    22:55,
    221:56,
    222:57,
    223:58,
    23:59,
    233:60,
    234:61,
    235:62,
    236:63,
    238:64,
    24:65,
    240:66,
    245:67,
    246:68,
    25:69,
    254:70,
    26:71,
    266:72,
    267:73,
    268:74,
    27:75,
    28:76,
    29:77,
    3:78,
    30:79,
    31:80,
    32:81,
    33:82,
    34:83,
    35:84,
    350:85,
    36:86,
    360:87,
    37:88,
    38:89,
    39:90,
    4:91,
    40:92,
    41:93,
    412:94,
    42:95,
    420:96,
    421:97,
    427:98,
    429:99,
    43:100,
    432:101,
    44:102,
    45:103,
    48:104,
    497:105,
    498:106,
    5:107,
    50:108,
    51:109,
    516:110,
    517:111,
    518:112,
    523:113,
    526:114,
    53:115,
    54:116,
    55:117,
    555:118,
    56:119,
    57:120,
    58:121,
    59:122,
    6:123,
    60:124,
    61:125,
    62:126,
    63:127,
    64:128,
    67:129,
    68:130,
    69:131,
    7:132,
    711:133,
    72:134,
    74:135,
    75:136,
    76:137,
    77:138,
    777:139,
    78:140,
    79:141,
    799:142,
    8:143,
    80:144,
    800:145,
    81:146,
    82:147,
    83:148,
    84:149,
    85:150,
    86:151,
    875:152,
    876:153,
    887:154,
    888:155,
    89:156,
    893:157,
    895:158,
    897:159,
    9:160,
    90:161,
    901:162,
    902:163,
    91:164,
    910:165,
    92:166,
    950:167,
    96:168,
    98:169,
    99:170,
}

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