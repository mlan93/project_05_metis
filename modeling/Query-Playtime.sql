/* Query below used to pull the playtime for all game-user combinations with 5 or more minutes of gameplay to
   remove cases where minutes recorded are related to installation time. */

CONNECT steam;
SELECT steamid, appid, playtime_forever FROM Games_2 WHERE playtime_forever >= 5;
