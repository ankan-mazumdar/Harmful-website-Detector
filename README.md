# Automated_Data_Labelling_of_URL
Deployed URL link : https://automated-url-labelling.herokuapp.com/

FUNCTIONING OF WEB APPLICATION

On launching the URL in browser, the initial page shows the instructions to operate the
page.

1. Just browse and select a csv file having URL data and click on predict button to get out baseline model predictions.
2. On predicting, we will get the data along with predictions and confidence scores against each prediction. Just validate of the predictions are correct or not. In case of any incorrect predictions, just correct the predicted label and click on save changes.
3. On clicking the save changes button a dataframe will appear with the modified values and it will also get saved in our data folder.
4. Once we have our modified data saved, we can retrain the baseline model using the new data. We have used Logistic Regression with warm start as True which lets us train the model from where we left at and not retraining from scratch. This helps to reduce computational cost and is fast as well.
5. After retraining, our baseline model will be modified with the newly trained model and over the next iterations, our model’s performance will improve.

We have used LIME to showcase our model interpretability
