# Automated_Data_Labelling_of_URL

FUNCTIONING OF WEB APPLICATION

On launching the URL in browser, the initial page shows the instructions to operate the
page.

1. Just browse and select a csv file having URL data and click on predict button to get out baseline model predictions.

![image](https://user-images.githubusercontent.com/69012134/210370282-f713c0e9-be6c-4466-a0de-a56910267bef.png)


2. On predicting, we will get the data along with predictions and confidence scores against each prediction. Just validate of the predictions are correct or not. In case of any incorrect predictions, just correct the predicted label and click on save changes.

![image](https://user-images.githubusercontent.com/69012134/210370622-b989510b-88b9-4dcb-ac34-5937e56f070e.png)

3. On clicking the save changes button a dataframe will appear with the modified values and it will also get saved in our data folder.

![image](https://user-images.githubusercontent.com/69012134/210370705-8a9311fc-a4a0-4cdf-8f01-56304bf691e3.png)

4. Once we have our modified data saved, we can retrain the baseline model using the new data. We have used Logistic Regression with warm start as True which lets us train the model from where we left at and not retraining from scratch. This helps to reduce computational cost and is fast as well.

5. After retraining, our baseline model will be modified with the newly trained model and over the next iterations, our model’s performance will improve.

We have used LIME to showcase our model interpretability

![image](https://user-images.githubusercontent.com/69012134/210370786-a25cee43-29e0-4587-a5de-4eaad9fadd75.png)

