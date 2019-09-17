# Project 05 - Video Game Recommendation System
**By Max Lan**

**Overview:**

The repository contains two folders: modeling and flask_app. Reviewer should begin with the modeling folder, then examine the flask_app folder

modeling folder: Data was pulled from the MySQL database hosted on RDS using Query-Playtim.sql into playtime_5_min.csv (which only contains 20,000 rows of sample data due to file size limits on Github). Model_CV.py was used to performed dross validation and gridsearch for the ALS recommendation model, while Final_Model.py was used to generate the final model trained on the training and validation set after grid search determined the optimal parameters.

flask_app folder: contains the code for the flask app using the model generated in the modeling folder. This model was kept in the final_model subfolder (excluded due to file size limits on Github). The template contains the app template used for recommendation, recommender_api.py includes thre functions used, and recommender_app.py calls the form and performs the recommendations after impoting functions from recommender_apy.py.

The project presentation slides are also kept in the main github, named Project_05_Presentation.pdf.
