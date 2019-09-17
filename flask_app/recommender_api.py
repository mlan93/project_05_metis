"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.

This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import requests
import time
import pandas as pd
import pyspark
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql import Row
from pyspark.sql import functions as func

# Initialize spark
spark = (pyspark.sql.SparkSession
         .builder
          .config('spark.executor.memory', '10g')
         .config('spark.driver.memory', '10g')
         .config('spark.driver.maxResultSize', '10g')
         .getOrCreate() 
        )

# Load recommendation model
model_name = '../Modeling_Base-Hours/final_model'

# RMSE of final model: 1.0425757964388849
model = ALSModel.load(model_name)

def load_id(steam_id_dict):
    '''
    Takes the int64 Steam ID entered per the form and converts it to a spark dataframe with the int64 and int32
    versions of the ID.

    Parameters: steam_id_dict is the int64 Steam ID input by the user.

    Output: a spark dataframe with the int64 Steam ID in one column and int32 in the other.
    '''
    
    steam_id = int(steam_id_dict['steam_id'])
    
    # convert user_id to int32 (to ensure compatibility with model results)
    steamid_int = steam_id - 76561197960265729

    # Convert input to spark dataframe for transformation
    id_list = [(steam_id, steamid_int)]
    col_names = ["steamid", "steamid_int"]

    id_df = spark.createDataFrame(id_list, col_names)
    
    return id_df

def recommend(steam_id_dict, model = model, n = 100):
    '''
    Generates the top 100 recommended games for the Steam ID using the trained ALS model.

    Parameters: steam_id_dict is the form input from the template (in the form of a dictionary), model is the
                previously trained ALS model, n is the number of recommended games per user.

    Output: a list of n recommendations (integer app IDs)
    '''
    user_df = load_id(steam_id_dict)
    result = model.recommendForUserSubset(user_df, n).select('recommendations')
    result_list = result.collect()[0][0]
    
    result_list = [result[0] for result in result_list]
    
    return result_list

def get_user_library(steam_id_dict):
    '''
    Obtains the user's current Steam library by requesting the data from the Steamworks API.

    Parameters: steam_id_dict is the form input from the template (in the form of a dictionary)

    Output: a list of the user's games in the form of integer app ID's and the user's steam id in int64 form.
    '''
    steam_id = int(steam_id_dict['steam_id'])
    
    # Load steam key for API access
    key = open('steam_key.txt', 'r').read()
    
    url = 'https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/'
    
    params = {'key': key, 'steamid': steam_id, 'include_played_free_games': 1}
    
    r = requests.get(url, params = params)
    
    data = r.json()['response']['games']
    
    library_list = [app['appid'] for app in data]
    
    return library_list, steam_id

def check_owned(result_list, library_list):
    '''
    Compares the model's recommended games list and the user's current library. Only games that are not in the
    user's current library are kep in the new recommendation list.

    Parameters: result_list is the output of the ALS model (a list of 100 game recommendations) and library_list
                is the ouput of the API request for the user's owned games. Both are lists of integers for the
                app IDs.

    Output: a list of recommendations (integer app IDs) not in the user's current Steam library.
    '''
    new_result_list = [x for x in result_list if x not in library_list]
    # Only keep 20 elements (10 more than necessary to account for games that were removed from the Steam platform
    # after the data was scraped).
    new_result_list = new_result_list[:20]
    
    print("new_result_list", new_result_list)
    
    return new_result_list

def get_app_info(app_id):
    '''
    Requests the information for the app, including the name, app ID, banner url, and other information for every app.

    Parameters: app_id is an integer of game's unique ID.

    Output: a dictionary containing the various information for each game.
    '''
    url = 'https://store.steampowered.com/api/appdetails?appids=' + str(app_id)
    header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10; rv:33.0) Gecko/20100101 Firefox/33.0',
        'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding':'gzip, deflate, sdch',
        'Accept-Language':'en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4,zh-TW;q=0.2'}
    
    r = requests.get(url, header)
    data=r.json()
    
    if data[str(app_id)]['success'] == True:
        data = data[str(app_id)]['data']
    
    else:
        data = 'FAIL'
    
    return data
   
def get_banner(data):
    '''
    Obtains the url for the game's banner image.

    Parameters: data is the dictionary containing information for each game.

    Output: a string url for the banner image.
    '''
    img_url = data['header_image']
    
    return img_url

def top_game_ids(n = 10):
    '''
    Obtains the app IDs for the top n games on Steam by playtime in the last 2 weeks from the SteamSpy API.

    Parameters: n is the number of results to pull.

    Output: a list of app_ids representing the top games by playtime in the last 2 weeks.
    '''
    url = 'https://steamspy.com/api.php?request=top100in2weeks'
    r = requests.get(url)
    data = r.json()
    
    top_list = list(data.keys())[:n]
    
    return top_list

def get_all_app_data(app_id_list):
    '''
    Obtains the game data for a list of games. Leverages the get_app_info function above.

    Parameters: a list of game app IDs.

    Output: a list of data for each game in dictionary form and a list of urls for each banner image.
    '''
    data_list = []
    banner_list = []
   
    for app_id in app_id_list:
        info = get_app_info(app_id)
        
        if info == 'FAIL':
            continue
        
        data_list.append(info)
#         time.sleep(0.2)
    
    for data in data_list:
        banner_list.append(get_banner(data))
    
    return data_list, banner_list
