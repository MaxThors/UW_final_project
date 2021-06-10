# Presentation Plan

Max: Slides 1-3 (2-3 minutes)
* New ice cream company looking for a new flavor to make
* Where do we start?
* We decided to look at customer reviews of four different companies and dozens of flavors
* Used NLP to parse through written-out reviews for deeper analysis
* Streamline product development and make it more efficient
* instead of going through so much back and forth with experimentation, we can be much more efficient with our time
* As I said earlier, we can make the product development process more efficient and cut out several steps

Kylie: Slides 4-6 (2-3 minutes)
Considerations while examining data-
  * These 4 brands were selected because they allowed for both positive and negative reviews on their websites whereas other companies only allowed 4 or 5 star reviews to be published
  * Does not take into account any review sensoring or fake reviews written by someone who had not actually consumed the product and wrote a review solely to help/harm overall rating
  * Should be noted, some reviews were written as a part of a promotion (similar to Amazon Vine Analysis project) but the exact details of different promotions are unknown
  * Looked at ice creams with a 4+ overall rating then filtered products table and created high_rating file
  * Using those products with a 4+ rating, filtered reviews file to contain only reviews of those products by joining high_rating and reviews files
  * Once reviews had been filtered, we began out machine learning process
  
Ashley: Slides 7-9 (2-3 minutes)

- After preprocessing the data, we performed Natural Language Processing NLP to acquire the words most associated with highest rated ice creams.
- NLP is the process of converting normal language to machine readable format.
- Data Retrieval: Here we take over 3,400 Amazon Reviews and perform the required step of Preprocessing the text data to prepare it for machine learning. We ran the NLP model to exclude standard stopwords then tokenize & normalize every word in the review. At this point, we were able to run Term Frequency-Inverse Document Frequency to evaluate how relevant each word is verses every other word in the reviews. 
- Data Classification:  For our use case to work, the computer must know how to classify our text.  We accomplished this by assigning reviews with a star rating of 4 or more as positive sentiment and 3 or less as negative sentiment.
- Information Extraction: Using NLP, we will extract certain aspects of the review text data to present the finding for our new specialty flavor. 
- “Bag of Words”: After seeing our initial results, we began feature engineering. We then developed a list of custom stopwords and ran the NLP model to tokenize & normalize every word in the review. We then collected the 1000 most common words and kept only the recipe and texture related words. A total of 127 words, our “Bag of Words” became the features used to train classifier models.
- “Feature Significance”: Now that we have our “Bag of Words,” We reran TF-IDF to re-capture the relevance of these words and rank their significance.
- “Random Forest”: We chose the Random Forest Model to help us predict the best specialty ice cream because of its ability to handle a lot of features without overfitting. It also ranks the importance of our features.
- Unfortunately, we learned the hard way that this won’t predict the best flavor for us and only predict if a review is positive or not.
- “VADER Sentiment Analyzer”: So Plan B  VADER sentiment analyzer. Running the VADER Sentiment Analyzer produced a sentiment score, better known as compound score, for every single review. Using this information, we were able to use the groupby function to get an average sentiment score based only on the review text for every ice cream in our dataset.  This worked out great because VADER agreed with our formula for positive sentiment 93% of the time.




Brian: Dashboard (2 minutes)
