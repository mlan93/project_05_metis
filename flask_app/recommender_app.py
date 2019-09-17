import flask
from flask import request
from recommender_api import *

# Initialize the app

app = flask.Flask(__name__)


# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!
@app.route("/")
def hello():
    return "It's alive!!!"


@app.route("/predict", methods=["POST", "GET"])
def predict():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)
    result_list = recommend(request.args, n = 100)
    library_list, steam_id = get_user_library(request.args)
    
    new_result_list = check_owned(result_list, library_list)
    
    data_list, banner_list = get_all_app_data(new_result_list)
    
    top_games_list = top_game_ids()
    top_data_list, top_banner_list = get_all_app_data(top_games_list)
    
    return flask.render_template('recommender.html', data_list = data_list, banner_list = banner_list, \
                                 top_data_list = top_data_list, top_banner_list = top_banner_list, steam_id = steam_id)

# @app.route("/game", methods=["POST", "GET"])
# def game():
#     # request.args contains all the arguments passed by our form
#     # comes built in with flask. It is a dictionary of the form
#     # "form name (as set in template)" (key): "string in the textbox" (value)
#     data = get_app_info(request.args['app_id']
    
#     top_games_list = top_game_ids()
#     top_data_list, top_banner_list = get_top_games(top_games_list)
    
#     return flask.render_template('game.html', )

# Start the server, continuously listen to requests.
# We'll have a running web app!

# For local development:
app.run(debug=True)

# For public web serving:
# app.run(host='0.0.0.0')
