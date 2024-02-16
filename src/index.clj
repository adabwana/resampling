(ns index)

;# Assignment 3: Resampling, Selection, & Regularization
;
;Instructions:
;
;* Select a dataset suitable to regression at your discretion
;* Using built in functionality, apply the following algorithms to your data using 5 fold cross validation and RMSE:<br>
  ;&nbsp;&nbsp;&nbsp;&nbsp;*Lasso Regression<br>
  ;&nbsp;&nbsp;&nbsp;&nbsp;*Ridge Regression<br>
  ;&nbsp;&nbsp;&nbsp;&nbsp;* Linear or SGD Regression w/ KBestFeature Selection<br>

;All final code should be in the main.py file.
;Modify the "README.md" file to include the following sections:<br>
;&nbsp;&nbsp;&nbsp;&nbsp;* **Summary**: Recap what was learned in class. This should be thorough enough that someone else could understand what the class was about <br>
;Questions (Answer these):<br>
;What were the best hyperparameters for Lasso/Ridge Regression?<br>
;Which features did Lasso/Ridge Regression choose?<br>
;What was the optimal K in K Best Features Selection?<br>
;Make sure that your README file is formatted properly and is visually appealing. It should be free of grammatical errors, punctuation errors, capitalization issues, etc. Sentences should be complete.
;
;What I did:
;
;* Selected [Boston Housing Prices](https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data?resource=download). One of my favorite regression datsets to play around with and teach.
;* Looked at the data with different transformations in mind. Standardizing and Tukey's ladder transformation.
;* Implemented regularization techniques--ridge and lasso--and created a best subset model of the 13 regressors.
;* Compared fitting model to raw vs fitting model to Tukey tranformed data.