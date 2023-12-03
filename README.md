# Talent Scouting Classification with Machine Learning
![istockphoto-500240235-612x612](https://github.com/YaseminOzturkk/scoutium_talenter_hunting/assets/48058898/1c907749-4e79-4f60-92be-1375b8270351)


This GitHub repository contains a machine learning model developed to classify talent scouting based on player attributes and ratings provided by scouts from the Scoutium football monitoring platform.

## Business Problem
Predicting the classification (average, highlighted) of players based on ratings given to their attributes by scouts monitoring them during matches.

## Dataset Story
The dataset consists of information gathered from football players observed during matches by Scoutium. It includes evaluations by scouts on various player attributes along with the assigned ratings.

## Requirements
The Python libraries and models used in this project include:

* Pandas
* Seaborn
* Matplotlib
* Scikit-learn
* LightGBM

## Data Preprocessing and Feature Engineering
The project includes various data preprocessing and feature engineering steps:

* Merging the dataset
* Creating a pivot table
* Exploratory Data Analysis (EDA)
** EDA steps include:

*** Overview of the dataset
*** Analysis of categorical and numerical variables
*** Correlation analysis
*** Investigating the relationship between the target variable and other features
* Handling missing values and outliers
* Label encoding for categorical variables
* Scaling numerical variables

# Machine Learning Model
A LightGBM (Gradient Boosting) model has been developed to predict the potential labels of players. The model's performance is evaluated using metrics such as ROC-AUC, F1 Score, Precision, Recall, and Accuracy.

# Model Inspection and Important Variables
The project also includes functions to evaluate the model's performance and visualize important variables.

# Usage
You can clone this repository to your local machine or download it as a ZIP file to explore and continue development.

Note: The dataset and file paths should be updated.

We hope this project inspires you and proves to be useful in your endeavors!🌟
