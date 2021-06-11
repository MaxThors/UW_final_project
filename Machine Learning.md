# Machine Learning

For this project we will be developing an unsupervised machine learning model to help us determine if a new ice cream flavor will have a high rating. We will have already filtered the initial dataset to contain only products with high ratings (>=4), and examine the text of the reviews of the high ratings data. Using NLP, we will identify key words in the text and the data will then be clustered together by common themes.

Because of the filtering, the model will be using the whole dataset as input, which is why an unsupervised model was chosen.

The initial preprocessed data containing products with high ratings can be located in the Resources folder. Anything with a rating greater than or equal to 4.0 was placed into a new dataframe (high_rating_df). The "subhead" column was dropped from this dataset because only Ben & Jerry's brand contained a subhead. Additional columns may be dropped during segment 2.

Once the NLP process is complete, it will be decided if the data needs to be scaled. We will also need to convert the relevant string columns to a numerical format and drop ones that can't be used.

Initiallly, we decide to use a Random Forest Model to help predict the best specialty ice cream because of its ability to handle a lot of features without overfitting. It also ranks the importance of our features. Unfortunately, we found this model doesn’t predict the best flavor, instead only predicting if a review is positive or negative.

Because of the Random Forest Model shortfalls, we chose to use a VADER sentiment analyzer. VADER produced a sentiment score, also referred to as a compound score, for each review of the previously filtered products with high ratings (>=4). Using this information, we were able to use the groupby function to get an average sentiment score based only on the review text for every ice cream in our dataset. This worked well because VADER agreed with our formula for positive sentiment 93% of the time.
