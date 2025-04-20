import requests
import pandas as pd
import time
import re
from dotenv import load_dotenv
import os
from champion_dictionary import Champion_to_Id, Id_to_Champion

def get_league_leaderboard_url(riot_api_key, region, league, queue_type):
    '''
    DESCRIPTION:
        Obtain leaderboard API URL based on the region, league, and queue type
    
    INPUTS:
        riot_api_key (str):     riot api key from developer portal
        region (str):           na1,br1,eun1,euw1,jp1,kr,la1,la2,me1,oc1,ru,sg2,tr1,tw2,vn2
        league (str):           challengerleagues,grandmasterleagues,masterleagues
        queue_type (str):       ranked_solo_5x5,ranked_solo_sr,ranked_solo_tt
    
    OUTPUTS:
        URL (str):              URL for API request to get leaderboard 
    '''
    return f'https://{region}.api.riotgames.com/lol/league/v4/{league}/by-queue/{queue_type}?api_key={riot_api_key}'

def get_matches_from_puuid_url(riot_api_key, region, player_puuid, match_type="ranked", num_matches=5):
    '''
    DESCRIPTION:
        Obtain # of match ids from player API URL based on the region, player, match type, and queue type
    
    INPUTS:
        riot_api_key (str):     riot api key from developer portal
        region (str):           americas,asia,europe,sea
        player_puuid (str):     player puuid
        match_type (str):       ranked,normal,tourney,tutorial
                                Filter the list of match ids by the type of match. This filter is mutually inclusive of the queue filter meaning any match ids returned must match both the queue and type filters.
        num_matches (int):      Defaults to 20. Valid values: 0 to 100. Number of match ids to return.
    
    OUTPUTS:
        URL (str):              URL for API request to get match IDs from player 
    '''
    return f'https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{player_puuid}/ids?type={match_type}&start=0&count={num_matches}&api_key={riot_api_key}'

def get_match_data_from_match_id_url(riot_api_key, region, match_id):
    '''
    DESCRIPTION:
        Obtain match data from match id API URL based on the region and match id
    
    INPUTS:
        riot_api_key (str):     riot api key from developer portal
        region (str):           americas,asia,europe,sea
        match_id (str):         challengerleagues,grandmasterleagues,masterleagues
    
    OUTPUTS:
        URL (str):              URL for API request to get match IDs from player 
    '''
    return f'https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={riot_api_key}'

def try_request(api_url, description, seconds_to_retry):
    '''
    DESCRIPTION:
        Tries to get a response from a get request, will retry up to 20 times with 
    
    INPUTS:
        api_url (str):          API IRL to get response from
        description (str):      For help text to describe what this request is for
        retry_after (int):      Number of seconds to retry the request
    '''
    retry_count = 0
    max_retries = 20

    while retry_count <= max_retries:
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"Rate limited. Sleeping for {seconds_to_retry} seconds. Retry count: {retry_count}")
                time.sleep(seconds_to_retry)
            else:
                print(f"Failed to fetch {description}: HTTP {response.status_code}")
                break
        except requests.exceptions.Timeout:
            print(f"Timeout occurred while fetching {description}. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Request error for {description}: {e}")
            break 
        
        retry_count += 1
        time.sleep(1)  # brief pause before retrying
    return None

def read_txt_to_set(file_path):
    '''
    DESCRIPTION:
        Reads txt file and return a set 
    
    INPUTS:
        file_path (str):        File path to txt file
    
    OUTPUTS:
        Output (set(str)):           Set containing a read of each line in txt or empty set
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()

def append_to_txt(item,file_path):
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

def grab_leaderboard(riot_api_key=None, count=None, queue_type='RANKED_SOLO_5x5', region="na1",
                     leagues=['challengerleagues','grandmasterleagues','masterleagues'], file_path=f"leaderboard_df.xlsx"):
    '''
    DESCRIPTION:
        Grabs top x amount of players and returns a configured df with columns "rank","summonerId","puuid","leaguePoints","wins","losses","hotStreak"
    
    INPUTS:
        riot_api_key (str):     riot api key from developer portal
        count (int):            Number of players from the top to grab
        region (str):           na1,br1,eun1,euw1,jp1,kr,la1,la2,me1,oc1,ru,sg2,tr1,tw2,vn2
        leagues (list(str)):    challengerleagues,grandmasterleagues,masterleagues
        file_path (str):        File path to txt save leaderboard dataframe
    
    OUTPUTS:
        Output (type):          Configured leaderboard df
    '''
    for league in leagues:
        players_api_url = get_league_leaderboard_url(riot_api_key, region, league, queue_type)
        
        response = try_request(players_api_url, "players from leaderboard", 10)
            
        if response:  
            leaderboard_df = pd.DataFrame(response.get('entries', []))
            leaderboard_df = configure_leaderboard(leaderboard_df)
            leaderboard_df.to_csv(file_path, mode='a', sep='\t', header=True, index=False)
            
        if count and len(leaderboard_df) > count:
            break
        
    return leaderboard_df.head(count) if count else leaderboard_df

def configure_leaderboard(leaderboard):
    '''
    DESCRIPTION:
        Removes 'rank','veteran','inactive','freshBlood' from the response json
        Return df with columns of 'rank','summonerId','puuid','leaguePoints','wins','losses','hotStreak'
    
    INPUTS:
        leaderboard (df):          dataframe containing top players
    
    OUTPUTS:
        leaderboard (df):          dataframe containing top players
    '''
    '''
    leaderboard (input) - string:      leaderboard to make edits to columns
    '''
    leaderboard = leaderboard.sort_values('leaguePoints',ascending=False)
    leaderboard = leaderboard.drop(columns=['rank','veteran','inactive','freshBlood'])
    leaderboard = leaderboard.reset_index()
    leaderboard = leaderboard.rename(columns={'index':'rank'})
    leaderboard['rank'] += 1
    return leaderboard

def collect_all_matches(riot_api_key=None, leaderboard_df=None, region=None, match_type='ranked', number_matches=5,
                        file_path_for_processed_players="processedPlayerPUUIDs.xlsx", file_path_for_match_ids="matches_df.xlsx"):
    '''
    DESCRIPTION:
        Grabs top players from leaderboard, collects all unique match ids, and stores into a txt file
    
    INPUTS:
        riot_api_key (str):        riot api key from developer portal
        leaderboard (df):          leaderboard of players that contain player puuid to collect matches from
        region (df):               americas,asia,europe,sea
        match_type (str):          only collect matches of this type (ranked,normal,tourney,tutorial)
        number_matches (int):      number of matches to collect from each player
        file_path_for_processed_players (str):      file path to append processed players
        file_path_for_match_ids (str):              file path to save match ids
    
    OUTPUTS:
        all_match_ids (set(str)):  set containing all unique match ids from all top players
    '''
    all_match_ids = read_txt_to_set(file_path_for_match_ids)
    processed_players = read_txt_to_set(file_path_for_processed_players)
    count_match = 0
    count_player = len(processed_players)
    for puuid in leaderboard_df['puuid']:
        if puuid in processed_players:
            continue
        matches = collect_matches_from_player(riot_api_key, region, puuid, match_type, number_matches)

        for match_id in matches:
            if match_id not in all_match_ids:
                count_match += 1
                print(f"match_id count: {count_match}")
                append_to_txt(match_id, file_path_for_match_ids)

        all_match_ids.update(matches)
        count_player += 1
        print(f"\n player count: {count_player}\n")
        append_to_txt(puuid, file_path_for_processed_players)
        processed_players.add(puuid)

    return all_match_ids

def collect_matches_from_player(riot_api_key=None, region=None, puuid=None, match_type='ranked', num_matches=5):
    '''
    DESCRIPTION:
        Collects # of matches from player and stores in a set
    
    INPUTS:
        riot_api_key (str):        riot api key from developer portal
        region (str):              americas,asia,europe,sea
        puuid (str):               Unique player puuid
        match_type (str):          Only collect matches of this type (ranked,normal,tourney,tutorial)
        num_matches (int):         Number of matches to grab
    
    OUTPUTS:
        match_ids (set(str)):      Set of match ids
    '''
    match_ids = set()
    matches_api_url = get_matches_from_puuid_url(riot_api_key,region,puuid,match_type,num_matches)

    response = try_request(matches_api_url,"matches from player",10)
    if response:
        match_ids.update(response)
            
    return match_ids

def obtain_match_data(riot_api_key, all_match_ids, region, file_path_for_processed_matches, file_path_for_data):
    '''
    DESCRIPTION:
        Gathers data from each match and stores into a txt file (file_path_for_data)
    
    INPUTS:
        riot_api_key (str):        riot api key from developer portal
        all_match_ids (set(str)):  set of all match ids to gather data
        region (str):              americas,asia,europe,sea
        file_path_for_processed_matches (str):      file path to save processed match ids
        file_path_for_data (str):                   file path to store data [patch, team1, team2] where team1 is losing team and team2 is winning team
    '''
    all_proccessed_matches_set = read_txt_to_set(file_path_for_processed_matches)
    
    for match_id in all_match_ids:
        if match_id not in all_proccessed_matches_set:
            match_json = get_match_json(riot_api_key,region,match_id)
            data = process_match_json_per_team(match_json)
            append_to_txt(data,file_path_for_data)
            append_to_txt(match_id,file_path_for_processed_matches)
            all_proccessed_matches_set.add(match_id)
    return

def get_match_json(riot_api_key, region,match_id):
    '''
    DESCRIPTION:
        Obtains json of match from riot api
    
    INPUTS:
        riot_api_key (str):        riot api key from developer portal
        region (str):              americas,asia,europe,sea
        match_id (type):           Match id
    
    OUTPUTS:
        response (json):           response.json()
    '''
    
    match_data_api_url = get_match_data_from_match_id_url(riot_api_key, region, match_id)
    response = try_request(match_data_api_url,"match data from match id",10)

    return response

def process_match_json_per_team(match_json):
    '''
    DESCRIPTION:
        Format's match json into the format [patch, team1, team2] where team1 is losing team and team2 is winning team
    
    INPUTS:
        match_json (json):         match data json
    
    OUTPUTS:
        data (array):              data in array or none 
    '''
    if not match_json or 'metadata' not in match_json:
        print("Invalid match_json structure:", match_json)
        return None
    info = match_json['info']
    players = info['participants']
    patch = re.match(r"(\d+\.\d+)", info['gameVersion'])
    patch = int(float(patch.group(1))*100)


    data = [patch]
    team1 = process_player_champion_name_to_id(players, start_index=0, end_index=4, use_champion_name=False)
    team2 = process_player_champion_name_to_id(players, start_index=5, end_index=9, use_champion_name=False)

    if players[0]['win']:
        data = data + team2 + team1
    else:
        data = data + team1 + team2

    return data

def process_player_champion_name_to_id(players, start_index, end_index, use_champion_name):
    '''
    DESCRIPTION:
        From players json, converts champion name into id if champion_name is True, else obtain champion name
    
    INPUTS:
        players (list(str)):       Array of player puuids
        start_index (int):         Start index to process
        end_index (int):           end index from players array
        champion_name (bool):      True: champion name
                                   False: champion id
    
    OUTPUTS:
        player_champs (list):      array of player champion name or id
    '''
    player_champs = []
    
    if use_champion_name:
        for i in range(start_index, end_index+1):
            player_champs.append(Id_to_Champion[players[i]['championId']])
            return player_champs
        
    for i in range(start_index, end_index+1):
        player_champs.append(players[i]['championId'])
    
    return player_champs
    
def main():
    load_dotenv()
    riot_api_key = os.environ.get('riot_api_key')
    if riot_api_key is None:
        raise ValueError("riot_api_key not found. Make sure your .env file is set correctly.")
    
    leaderboard_df_file_path = "leaderboard_df.txt"
    all_match_ids_file_path = "all_match_ids.txt"
    processed_player_puuids_file_path = "processed_player_puuids.txt"
    processed_match_ids_file_path = "processed_match_ids.txt"
    data_file_path = "training_data.txt"
    region_leaderboard = "kr"
    region = "asia"
    
    if not os.path.exists(leaderboard_df_file_path):
        open(leaderboard_df_file_path, 'w').close()
    if not os.path.exists(all_match_ids_file_path):
        open(all_match_ids_file_path, 'w').close()
    if not os.path.exists(processed_player_puuids_file_path):
        open(processed_player_puuids_file_path, 'w').close()

    leaderboard_df = grab_leaderboard(riot_api_key, 
                                      count=5,
                                      queue_type='RANKED_SOLO_5x5',
                                      region=region_leaderboard,
                                      leagues=['challengerleagues','grandmasterleagues'],
                                      file_path=leaderboard_df_file_path)

    all_matches = collect_all_matches(riot_api_key, 
                                      leaderboard_df, 
                                      region,
                                      match_type='ranked', 
                                      number_matches=5,
                                      file_path_for_processed_players=processed_player_puuids_file_path,
                                      file_path_for_match_ids=all_match_ids_file_path)

    obtain_match_data(riot_api_key, all_matches, region, processed_match_ids_file_path, file_path_for_data=data_file_path)
        

main()