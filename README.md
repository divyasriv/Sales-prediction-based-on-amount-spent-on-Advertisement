# Sales-prediction-based-on-amount-spent-on-Advertisement

### Description of problem statement :
#### Given a dataset containing amount spent on platforms (TV,Newspaper,Radio) predict the sales.
### Solution method:
#### For this dataset tried out different regression models like
* Linear Regressor
* Decision Tree
* Random Forest
* Lasso
* Ridge
* XG Boost 
* Light Gradient Boosting.
#### Found scores for all models from which Random Forest,XG Boost and Light GBM gave best results.
#### Scores after Hyper Parameter tuning :
*For Random Forest*
##### Out[272]: 0.971835615937762
*For XG Boost*
##### Out[275]: 0.9786290210166928
*For Light Gradient Boosting without hyper paramter tuning*
##### Out[276]: 0.9846795572131498

### Deployment:
#### For deployment I used Object Oriented methodology.
#### In **process_data.py** models are loaded from script and input data is provided to them for predicton.
#### **app.py**  file uses Flask for web application development.
#### **index.html** and **results.html** are rendered templates for API.
