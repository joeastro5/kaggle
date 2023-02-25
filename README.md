# kaggle
Some data science projects on Kaggle

heart-failure-prediction-88-f1-score-with-xgboost.ipynb: The project is from this dataset on Kaggle: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data

From the description by the authors:
"Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help."

The dataset has 12 features from each patient, with the target "DEATH_EVENT" being whether or not the patient died of heart failure. After some data exploration and visualization, I try some feature engineering to make the data less skewed. Then, I apply some ML models (Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Random Forest, and XGBoost), along with Grid Search cross-validation for hyperparameter tuning. XGBoost does the best job, giving 88% averaged f1 score, recall, and precision.
