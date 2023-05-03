# Kaggle
Some data science projects on Kaggle.

# Forest Fires and Random Forests
[Forest Fires and Random Forests](https://github.com/joeastro5/kaggle/blob/main/forest-fires-and-random-forests.ipynb): The dataset comes from the UCI ML repository: https://archive.ics.uci.edu/ml/datasets/forest+fires (and the Kaggle link: https://www.kaggle.com/code/joewraga/forest-fires-and-random-forests).  This is a regression problem using 11 features to predict the area of damage (in hectares) done to forest fires in Northeast Portugal. A paper by P. Cortez and A. Morais describing and tackling the problem is here: https://core.ac.uk/download/pdf/55609027.pdf (P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimarães, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9).

Some notes from the paper:

* Some of the features are related to fuel codes: FFMC refers to moisture content surface litter and influences ignition and fire spread (ranges from 18.7 to 96.20). DMC (1.1 to 291.3) and DC (7.9 to 860.6) are the moisture content of shallow and deep organic layers. ISI (0.0 to 56.10) is a score that correlates with fire velocity spread. RH is relative humidity in percentage (15.0 to 100). Temperature is in Celsius (2.2 to 33.30), wind speed is in km/h (0.40 to 9.40), and rain is in mm/m2 (0.0 to 6.4).

* Data were collected from January 2000 to December 2003.

* The values denote instant records, as given by station sensors when the forest fire in question was detected (except for the rain variable, which denotes the accumulated precipitation within the previous 30 minutes).

* They find that spatial coordinates (X, Y) and time features are not as important.

* Best RMSE: 63.7, Best MAD: 12.7. These are the metrics I'll be using to evaluate ML models.

* An area of zero actually means an area less than  100𝑚2
  was burned (note, a hectare is  104𝑚2
 ).

* Of the weather conditions, the outside temperature is the most important feature, followed by the accumulated precipitation.

The target (area) is heavily skewed toward low values, so the logarithmic transformation is used.

Metrics used to evaluate models are RMSE and MAE. With RandomForest, I find RMSE=64.3 and the MAE=13.83, comparable to the results of Cortez & Morais (RMSE=63.7, MAE=12.7). The most relevant features are Temperature, DMC, X, Y, ISI, and DC.


# Heart Failure Classification
[Heart Failure Classification](https://github.com/joeastro5/kaggle/blob/main/heart-failure-prediction-88-f1-score-with-xgboost.ipynb): The project is from this dataset on Kaggle: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data

From the description by the authors:
"Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help."

The dataset has 12 features from each patient, with the target "DEATH_EVENT" being whether or not the patient died of heart failure. After some data exploration and visualization, I try some feature engineering to make the data less skewed. Then, I apply some ML models (Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Random Forest, and XGBoost), along with Grid Search cross-validation for hyperparameter tuning. XGBoost does the best job, giving 88% averaged f1 score, recall, and precision.
