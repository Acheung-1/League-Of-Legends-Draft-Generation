import requests
import pandas as pd
import time
import re
from dotenv import load_dotenv
import os
load_dotenv()
riot_api_key = os.environ.get('riot_api_key')


# Dictionaries to translate champion name and ID
Champion_to_Id = {
    "None":0,
    "Annie":1,
    "Kayle":10,
    "Xerath":101,
    "Shyvana":102,
    "Ahri":103,
    "Graves":104,
    "Fizz":105,
    "Volibear":106,
    "Rengar":107,
    "Master Yi":11,
    "Varus":110,
    "Nautilus":111,
    "Viktor":112,
    "Sejuani":113,
    "Fiora":114,
    "Ziggs":115,
    "Lulu":117,
    "Draven":119,
    "Alistar":12,
    "Hecarim":120,
    "Kha'Zix":121,
    "Darius":122,
    "Jayce":126,
    "Lissandra":127,
    "Ryze":13,
    "Diana":131,
    "Quinn":133,
    "Syndra":134,
    "Aurelion Sol":136,
    "Sion":14,
    "Kayn":141,
    "Zoe":142,
    "Zyra":143,
    "Kai'Sa":145,
    "Seraphine":147,
    "Sivir":15,
    "Gnar":150,
    "Zac":154,
    "Yasuo":157,
    "Soraka":16,
    "Vel'koz":161,
    "Taliyah":163,
    "Camille":164,
    "Akshan":166,
    "Teemo":17,
    "Tristana":18,
    "Warwick":19,
    "Olaf":2,
    "Nunu":20,
    "Bel'Veth":200,
    "Braum":201,
    "Jhin":202,
    "Kindred":203,
    "Miss Fortune":21,
    "Ashe":22,
    "Zeri":221,
    "Jinx":222,
    "Tahm Kench":223,
    "Tryndamere":23,
    "Briar":233,
    "Veigo":234,
    "Senna":235,
    "Lucian":236,
    "Zed":238,
    "Jax":24,
    "Kled":240,
    "Ekko":245,
    "Qiyana":246,
    "Morgana":25,
    "Vi":254,
    "Zilean":26,
    "Aatrox":266,
    "Nami":267,
    "Azir":268,
    "Singed":27,
    "Evelyn":28,
    "Twitch":29,
    "Galio":3,
    "Karthus":30,
    "Cho'Gath":31,
    "Amumu":32,
    "Rammus":33,
    "Anivia":34,
    "Shaco":35,
    "Yuumi":350,
    "Dr. Mundo":36,
    "Samira":360,
    "Sona":37,
    "Kassadin":38,
    "Irelia":39,
    "Twisted Fate":4,
    "Janna":40,
    "Gangplank":41,
    "Thresh":412,
    "Corki":42,
    "Illaoi":420,
    "Rek'Sai":421,
    "Ivern":427,
    "Kalista":429,
    "Karma":43,
    "Bard":432,
    "Taric":44,
    "Veigar":45,
    "Trundle":48,
    "Rakan":497,
    "Xayah":498,
    "Xin Zhao":5,
    "Swain":50,
    "Caitlyn":51,
    "Ornn":516,
    "Sylas":517,
    "Neeko":518,
    "Aphelious":523,
    "Rell":526,
    "Blitzcrank":53,
    "Malphite":54,
    "Katarina":55,
    "Pyke":555,
    "Nocturne":56,
    "Maokai":57,
    "Renekton":58,
    "Jarvan IV":59,
    "Urgot":6,
    "Elise":60,
    "Orianna":61,
    "Wukong":62,
    "Brand":63,
    "Lee Sin":64,
    "Vayne":67,
    "Rumble":68,
    "Cassiopia":69,
    "Leblanc":7,
    "Vex":711,
    "Skarner":72,
    "Heimerdinger":74,
    "Nasus":75,
    "Nidalee":76,
    "Udyr":77,
    "Yone":777,
    "Poppy":78,
    "Gragas":79,
    "Ambessa":799,
    "Vladamir":8,
    "Pantheon":80,
    "Mel":800,
    "Ezreal":81,
    "Mordekaiser":82,
    "Yorick":83,
    "Akali":84,
    "Kennen":85,
    "Garen":86,
    "Sett":875,
    "Lilia":876,
    "Gwen":887,
    "Renata Glasc":888,
    "Leona":89,
    "Aurora":893,
    "Nilah":895,
    "K'Sante":897,
    "Fiddlesticks":9,
    "Malzahar":90,
    "Smolder":901,
    "Milio":902,
    "Talon":91,
    "Hwei":910,
    "Riven":92,
    "Naafiri":950,
    "Kog'Maw":96,
    "Shen":98,
    "Lux":99
}
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
    234:"Veigo",
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

def get_league_leaderboard_url(riot_api_key,region,league,queue_type):
    '''
    DESCRIPTION:
        Obtain leaderboard API URL based on the region, league, and queue type
    
    INPUTS:
        riot_api_key (str):     riot api key from developer portal
        region (str):           na1,br1,eun1,euw1,jp1,kr,la1,la2,me1,oc1,ru,sg2,tr1,tw2,vn2
        league (str):           challengerleagues,grandmasterleagues,masterleagues
        queue_type (str):       ranked_solo_5x5,ranked_solo_sr,ranked_solo_tt
    OUTPUTS:
        URL (str):              url for API request to get leaderboard 
    '''
    return f'https://{region}.api.riotgames.com/lol/league/v4/{league}/by-queue/{queue_type}?api_key={riot_api_key}'

def get_matches_from_puuid_url(riot_api_key,region,player_puuid,match_type="ranked",num_matches=5):
    '''
    DESCRIPTION:
        Obtain # of matches from player API URL based on the region, player, match type, and queue type
    INPUTS:
        riot_api_key (str):     riot api key from developer portal
        region (str):           (americas,asia,europe,sea)
        player_puuid (str):     player puuid
        match_type (str):       (ranked,normal,tourney,tutorial)
                                Filter the list of match ids by the type of match. This filter is mutually inclusive of the queue filter meaning any match ids returned must match both the queue and type filters.
        num_matches (int):      Defaults to 20. Valid values: 0 to 100. Number of match ids to return.
    OUTPUTS:
        URL (str):              url for API request to get match IDs from player 
    '''
    return f'https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{player_puuid}/ids?type={match_type}&start=0&count={num_matches}&api_key={riot_api_key}'

def get_match_data_from_match_id_url(riot_api_key,region,match_id):
    '''
    DESCRIPTION:
        Obtain # of matches from player API URL based on the region, player, match type, and queue type
    INPUTS:
        riot_api_key (input) - string:     riot api key from developer portal
        region (input) - string:           (americas,asia,europe,sea)
        match_id (input) - string:         (challengerleagues,grandmasterleagues,masterleagues)
    OUTPUTS:
        URL (str):              url for API request to get match IDs from player 
    '''
    return f'https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={riot_api_key}'

def try_request(api_url,description):
    '''
    api_url (input) - string:          api url to get response from
    description (input) - string:      For help text to describe what this request is for
    '''
    retry_count = 0
    max_retries = 10

    while retry_count <= max_retries:
        response = requests.get(api_url)
        if response.status_code == 200:
            response_json = response.json()
            return response_json  
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 10))
            print(f"Rate limited. Sleeping for {retry_after} seconds...")
            time.sleep(retry_after)
            retry_count += 1
        else:
            print(f'Failed to fetch {description}: HTTP {response.status_code}')
            break  
    return None

def append_to_df(df_to_append,file_path):
    '''
    df_to_append (input) - string:     dataframe to append
    file_path (input) - string:        file path to save dataframe to
    '''
    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            existing_df = pd.read_excel(file_path)
            startrow = existing_df.shape[0] + 1 
            df_to_append.to_excel(writer, index=False, header=False, startrow=startrow)
    else:
        df_to_append.to_excel(file_path, index=False)
    return 

def read_txt_to_set(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()

def append_to_txt(item,file_path):
    '''
    item (input) - string:             item to append
    file_path (input) - string:        file path to save item to
    '''
    with open(file_path, 'a') as f:
        f.write(str(item) + '\n')
    return

def grab_leaderboard(riot_api_key=None,count=None,queue_type='RANKED_SOLO_5x5',region="na1",leagues=['challengerleagues','grandmasterleagues','masterleagues'],file_path=f"leaderboard_df.xlsx"):
    '''
    riot_api_key (input) - string:     item to append
    count (input) - int:               file path to save item to
    region (input) - string:           region
    leagues (input) - list(string):    list of leagues
    file_path (input) - string:        file path to save leaderboard dataframe
    '''
    for league in leagues:
        players_api_url = get_league_leaderboard_url(riot_api_key,league,queue_type)
        
        response = try_request(players_api_url,"players from leaderboard")
            
        if response:  
            leaderboard_df = pd.DataFrame(response.get('entries', []))
            leaderboard_df = configure_leaderboard(leaderboard_df)
            leaderboard_df.to_csv(file_path, mode='a', sep='\t', header=True, index=False)
            
        if count and len(leaderboard_df) > count:
            break
        
    return leaderboard_df.head(count) if count else leaderboard_df

def configure_leaderboard(leaderboard):
    '''
    leaderboard (input) - string:      leaderboard to make edits to columns
    '''
    leaderboard = leaderboard.sort_values('leaguePoints',ascending=False)
    leaderboard = leaderboard.drop(columns=['rank','veteran','inactive','freshBlood'])
    leaderboard = leaderboard.reset_index()
    leaderboard = leaderboard.rename(columns={'index':'rank'})
    leaderboard['rank'] += 1
    return leaderboard

def collect_all_matches(riot_api_key=None,leaderboard_df=None,match_type='ranked',number_matches=5,file_path_for_processed_players="processedPlayerPUUIDs.xlsx",file_path_for_match_ids="matches_df.xlsx"):
    all_match_ids = read_txt_to_set(file_path_for_match_ids)
    processed_players = read_txt_to_set(file_path_for_processed_players)
    count_match = 0
    count_player = len(processed_players)
    for puuid in leaderboard_df['puuid']:
        if puuid in processed_players:
            continue
        matches = collect_matches_from_player(riot_api_key,puuid,match_type,number_matches)

        for match_id in matches:
            if match_id not in all_match_ids:
                count_match += 1
                print(f"match_id count: {count_match}")
                append_to_txt(match_id,file_path_for_match_ids)

        all_match_ids.update(matches)
        count_player += 1
        print(f"\n player count: {count_player}\n")
        append_to_txt(puuid,file_path_for_processed_players)
        processed_players.add(puuid)

    return all_match_ids

def collect_all_matches(riot_api_key=None,leaderboard_df=None,region=None,match_type='ranked',number_matches=5,file_path="matches_df.xlsx"):
    '''
    riot_api_key (input) - string:     item to append
    leaderboard (input) - dataframe:   leaderboard of players that contain player puuid to collect matches from
    match_type (input) - string:       only collect matches of this type
    number_matches (input) - int:      number of matches to collect from each player in the leaderboard
    file_path (input) - string:        file path to save leaderboard dataframe
    '''
    all_match_ids = set()
    count = 0
    for puuid in leaderboard_df['puuid']:
        matches = collect_matches_from_player(riot_api_key,region,puuid,match_type,number_matches)

        for match_id in matches:
            if match_id not in all_match_ids:
                count += 1
                print(f"match_id count: {count}")
                append_to_txt(match_id,file_path)

        all_match_ids = all_match_ids.union(matches)


    return all_match_ids

def collect_matches_from_player(riot_api_key=None,region=None,puuid=None,match_type='ranked',num_matches=5):
    match_ids = set()
    matches_api_url = get_matches_from_puuid_url(riot_api_key,region,puuid,match_type,num_matches)

    response = try_request(matches_api_url,"matches from player")
    if response:
        match_ids.update(response)
            
    return match_ids
