# Talent Scouting Classification with Machine Learning
This GitHub repository contains a machine learning model developed to classify talent scouting based on player attributes and ratings provided by scouts from the Scoutium football monitoring platform.

## Business Problem
Predicting the classification (average, highlighted) of players based on ratings given to their attributes by scouts monitoring them during matches.

## Dataset Story
The dataset consists of information gathered from football players observed during matches by Scoutium. It includes evaluations by scouts on various player attributes along with the assigned ratings.

###scoutium_attributes.csv Variables:
*task_response_id: A set of evaluations by a scout for all players in a team's squad in a match.
*match_id: The id of the relevant match.
*evaluator_id: The id of the evaluator (scout).
*player_id: The id of the relevant player.
*position_id: The id of the position played by the player in that match.
1: Goalkeeper
2: Center Back
3: Right Back
4: Left Back
5: Defensive Midfielder
6: Central Midfielder
7: Right Wing
8: Left Wing
9: Attacking Midfielder
10: Forward
*analysis_id: A set of feature evaluations by a scout for a player in a match.
*attribute_id: The id of each feature evaluated for players.
*attribute_value: The value (score) assigned by a scout to a player's specific feature.
###scoutium_potential_labels.csv Variables:
*task_response_id: A set of evaluations by a scout for all players in a team's squad in a match.
*match_id: The id of the relevant match.
*evaluator_id: The id of the evaluator (scout).
*player_id: The id of the relevant player.
*potential_label: The label indicating the final decision of a scout for a player in a match (target variable).

## Requirements
The Python libraries and models used in this project include:

Pandas
Seaborn
Matplotlib
Scikit-learn
LightGBM

## Data Preprocessing and Feature Engineering
The project includes various data preprocessing and feature engineering steps:

*Merging the dataset
*Creating a pivot table
*Exploratory Data Analysis (EDA)
**EDA steps include:

***Overview of the dataset
***Analysis of categorical and numerical variables
***Correlation analysis
***Investigating the relationship between the target variable and other features
*Handling missing values and outliers
*Label encoding for categorical variables
*Scaling numerical variables

#Machine Learning Model
A LightGBM (Gradient Boosting) model has been developed to predict the potential labels of players. The model's performance is evaluated using metrics such as ROC-AUC, F1 Score, Precision, Recall, and Accuracy.

#Model Inspection and Important Variables
The project also includes functions to evaluate the model's performance and visualize important variables.

#Usage
You can clone this repository to your local machine or download it as a ZIP file to explore and continue development.

Note: The dataset and file paths should be updated.

We hope this project inspires you and proves to be useful in your endeavors!
