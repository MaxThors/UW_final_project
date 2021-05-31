# UW_final_project
## Members: Brian Forth, Ashley Green, Max Thorstad, Kylie Wrenn

### Communication Protocols
For the duration of this project, Slack will be our primary communication vehicle. We've created a group chat there and so far it has proven sufficient to get feedback from one another and brainstorm.

We also have a standing weekly meeting at 7pm CST every Wednesday via Zoom, in addition to utilizing weekly class time to strategize next steps and perform our analysis.

### Presentation

We selected this topic to determine what ice cream flavor would be the best for our new ice cream company. For our new flavor, we want a specialty flavor. Our source data is several csv files from Kaggle that compiled flavors and customer reviews from four different ice cream companies. The flavor dataset includes the company, name of the flavor, and a description. The review dataset includes a product rating from 1-5, votes on whether or not other customers found each review helpful, and a written-out review. We will use Natural Language Processing (NLP) to parse out the written reviews in order to determine which words are most associated with higher ratings so we can better determine a flavor for our specialty ice cream. We're going to use this process to maximize efficiency in our product development process.

[Google Slides Presentation](https://docs.google.com/presentation/d/1o12XOvr3tCzIAZenm0lMi9tgW_xn3FhA8FbdcHNaUy4/edit#slide=id.gdc1ec10833_0_77)
=======
We selected this topic to determine what ice cream flavor would be the best for our new ice cream company. For our new flavor, we want a specialty flavor. Our source data is several csv files from Kaggle that compiled flavors and customer reviews from four different ice cream companies. The flavor dataset includes the company, name of the flavor, and a description. The review dataset includes an overall product rating from 1-5, votes on whether or not other customers found each review helpful, and a written-out review that includes an individual consumer star rating of their product experience ranging from 1-5. 

We will use Natural Language Processing (NLP) to parse out the written reviews in order to determine which words are most associated with higher ratings so we can better determine a flavor for our specialty ice cream. This process will help us finalize our decision on a new specialty flavor and maximize efficiency in our product development process.


## Machine Learning Model
### Natural Language Processing (NLP)

NLP is the process of converting normal language to a machine readable format, which allows a computer to analyze text as if it were numerical data.  

### Information Extraction & Text Classification


The primary goal of the nlp_feature_extraction_vectorizing.ipynb notebook is to pre-process large amounts of text data in order to prepare it for an NLP model that extract information and classifies text. 

- Extracting information: Many NLP tasks require the ability to retrieve specific pieces of information from a given document. We want to extract certain aspects of the review text data to present a compelling case for our speciality flavor.

- Classifying text: For many of the aforementioned use cases to work, a computer must know how to classify a given piece of text. We will classify our "Bag of Words" by positive and negative sentiment. 


### Scope of the notebook:
source: nlp_feature_extraction_vectorizing.ipynb


#### 1.  Data Inspection
#### 2.  Add Sentiment Feature to data set
#### 3.  Create Product Sentiment Reviews Dataset
#### 4.  Tokenization, Normalization & Custom Stopword Filtering
#### 5.  Extract the most common words
#### 6.  Create "Bag of Words" data set
#### 7.  Term Frequency-Inverse Document Frequency (TF-IDF)
#### 8.  Split the Data into Training and Testing
#### 9.  Balanced Random Forest Classifier

<br>
<br>


***1.  Data Inspection***

We are starting with a review dataset ("Resources/helpful_clean_reviews_combined.csv") that has been filtered down to ice cream products that have achieved an overall amazon rating of 4 stars or higher, joined on the "key" with consumer reviews that have been filtered down to those that received more helpful_yes votes than helpful_no votes.

![helpful_clean_reviews_combined](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/helpful_clean_reviews_combined.png)

Knowing the data is key to ensuring its compatible with any functions or methods required for the code to perform.
We know, off the cusp, that we have to teach the computer how to understand natural language by encoding text to numerical data that it can be taught to interpret.  

So lets pull out our pen and paper to note what data must be transformed.

![data_overview](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/data_overview.png)

It doesn't like strings or null values so lets identify any of those. 

![data_info](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/data_info.png)

![null_values](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/null_values.png)

Also, we will remove any duplicate data as it doesn't tell us anything new and may skew results. 

![duplicate_entries](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/duplicate_entries.png)

We can drop these duplicates now before our pre-processing begins.


<br>
<br>

***2.  Add Sentiment Feature to data set***

Here we assign a value of 1 to reflect positive sentiment. This consists of star rating greater than or equal to 4. Any review with a star rating less than 4 gets a value of 0 to reflect negative sentiment. Remember, star rating is the rating left by the individual reviewer. It is different than the overall rating presented by Amazon.

![add_sentiment](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/add_sentiment.png)

Number of positive reviews: 2,739
Number of negative reviews: 685

<br>
<br>


***3.  Create Product Sentiment Reviews Dataset***

location: "Resources/product_sentiment_reviews.csv"


<br>
<br>

***4.  Tokenization, Normalization & Custom Stopword Filtering***

Here is where all the magic of splitting the reviews into individual words, putting each word into lower case, lemmatizing each to its base form, removing punctuations and excluding stop words occurs.

We perform this step with the NLTK library as it is the most popular in education and research for NLP.  


Here are the dependencies:

```
# import the Tokenizer library
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer

# RegexpTokenizer will tokenize according to any regular expression assigned. 
# The regular expression r'\w+' matches any pattern consisting of one or more consecutive letters.
reTokenizer = RegexpTokenizer(r'\w+')



from nltk.corpus import stopwords
from string import punctuation
stop_words = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk import FreqDist
```

We begin by creating a DataFrame, df_tokenize, with the product_sentiment_reviews table created in step 3. 
(location: "Resources/product_sentiment_reviews.csv")

Then we extract each individual word from the review into a list, with special attention to filter out stopwords.
Note: There are comments within the code to explain each line's purpose.

``` 
# initialize list to hold words
all_words = []


for i in range(len(df_tokenize['text'])):
    # separate review text into a list of words
    tokens = reTokenizer.tokenize(df_tokenize['text'][i])
    
   
    df_tokenize['text'][i] = []
    
    # iterate through tokens
    for word in tokens:
        # lower the case of each word
        word = word.lower()
        # exclude stop words
        if word not in stop_words:
            
            # Lemmatize words into a standard form and avoid counting the same word more than once
            word = lemmatizer.lemmatize(word)
            # append to list of all_words
            all_words.append(word)
            # append to text column of dataframe for appropriate row
            df_tokenize['text'][i].append(word)
```

The model has extracted each individual word from the review text in a list called "all_words," which holds 6153 unique words total.


Where are the custom stopwords, you ask? At this time, we are focused on completing a working model by ensuring everything works. We will develop our list of custom stopwords as we begin to fully train the model and implement advanced feature extraction techniques. Stay tuned.

<br>
<br>

***5.  Extract the most common words***

Next, we extract our "Bag of Words," also known as "Most Common Words."
We are starting with 500 words to get an idea of the type of words we should normalize and filter out at with advanced feature exaction.

```
# Extract the most common words from the list of all_words.

from nltk import FreqDist

# sort all of the words in the all_words list by frequency count
all_words = FreqDist(all_words)
# Extract the 500 most common words from the all_words list
most_common_words = all_words.most_common(500)

# create a list of most common words without the frequency count
word_features = []
for w in most_common_words:
    word_features.append(w[0])

#print 
most_common_words
```

There are  500 unique words in the most common words list.
<br>
![most_common_words](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/most_common_words.png)
<br>
And here's a pretty wordcloud displaying some of our most common words. 
<br>
![most_common_words_wordcloud](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/most_common_words_wordcloud.png)


<br>
<br>

***6.  Create "Bag of Words" data set***

Now that the model has extracted the Bag of Words, we add the most common words from each review as a new column called bag_of_words. 
The model will iterate through the list of word_features (same as most common words, excluding the frequency count) and append to the dataset. We create new list as a form of a checkpoint so not to overwrite any prior work we've done. This is important to maintain the integrity of the data since the computer doesn't automatically re-run all cells as we make changes to the code. 



```
# create Bag of Words DataFrame
df_bagofwords = pd.DataFrame(df_tokenize)

# create column for bag of words
df_bagofwords['bag_of_words'] = ""

# iterate dataframe to populate bag of words column
for i in range(len(df_bagofwords['text'])):
    # initialize empty column    
    df_bagofwords['bag_of_words'][i] = []
    
    # iterate through df row by row
    for word in df_bagofwords['text'][i]:
        # if a word in 'text' is in the most common words
        # note: this is simply the "most_common_words" without the count column
        if word in word_features:
            # if it is, add it to the bag of words cell
            df_bagofwords['bag_of_words'][i].append(word)

```

![bag_of_words_dataset](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/bag_of_words_dataset.png)


Here is an example showing the beauty of all our work so far. 

![extraction_example](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/extraction_example.png)

The model has extracted each individual word from the text, filtering out stopwords in the NLTK stopword library and our bag of words are limited to the 500 most common words.  There are still words that we would like to add to stopwords to create our custom stopwords, but for now we are happy to see the model is working as designed.


<br>
<br>

***7.  Term Frequency-Inverse Document Frequency (TF-IDF)***

Term Frequency-Inverse Document Frequency (TF-IDF) statistically ranks the words by importance compared to the rest of the words in the text. This is also when the words are converted from text to numbers.

Decision trees such as Random Forests are insensitive to monotone transformations of input features. Since multiplying by the same factor is a monotone transformation, TF-IDF is compatible with such models.  This is great as we are pre-processing for Random Forests, but the data is ready for other classifiers that may require TF-IDF as well. 

Let's take a look at the code to complete TF-IDF.

```
# import dependencies
from sklearn.feature_extraction.text import TfidfVectorizer

# create new DataFrame to hold encoded values 
df_tfidf_text = pd.DataFrame(df_bagofwords)

# convert text list to string and create string column
# Required for vectorizer. Running on a list will yield an error.
df_tfidf_text['bag_of_words_str'] = df_tfidf_text['bag_of_words'].apply(lambda x: ','.join(map(str, x)))

```


Our bag of words are now in string format, enabling the TfidVectorizer to analyze each individual word.

![bag_of_words_str](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/bag_of_words_str.png)


The TfidVectorizer now analyzes each individual word in our bag of words, filters out stop words via stemming. We allow this with tfidf as it groups all words which are derived from the same stem (i.e. delayed, delay), this grouping will increase the occurrence of this stem because frequencies are calculated using stem not words. It appears to perform better than the NLTK stopwords library, which again, we will investigate futher when training the model to achieve desired accuracy. 


```
# get 'text' term frequencies weighted by their relative importance (IDF)
tfidf = TfidfVectorizer(analyzer='word', stop_words = 'english')

# create variable to hold independent features and TFIDF
x = df_tfidf_text['bag_of_words_str']


# Fit and transform independent features
xtfidf = tfidf.fit_transform(x)

# Create encoded TFIDF vector for bag of words
tfdif_bagOfWords_df = pd.DataFrame(data = xtfidf.toarray(),
                        # set column header as feature names           
                        columns = tfidf.get_feature_names())
                        
# Rank top 20 terms from TFID in order of signifigance score
terms = tfidf.get_feature_names()

# sum tfidf frequency of each term through documents
sums = xtfidf.sum(axis=0)

# connecting term to its sums frequency
data = []
for col, term in enumerate(terms):
    data.append( (term, sums[0,col] ))

ranking = pd.DataFrame(data, columns=['term','rank'])
term_rank = ranking.sort_values('rank', ascending=False)

term_rank[:20]                        

```

And viola, we can now confirm that the model is statistically ranking the bag of words by importance.

![significance_rank](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/significance_rank.png)

Next, we merge the encoded TFIDF vector we created for our bag of words with the original columns we'd like to keep. 

![merge_encoded](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/merge_encoded.png)


<br>
<br>

***8.  Split the Data into Training and Testing***

Here we define our training and testing data in preparation for the Random Forest Classifier model. 
Sentiment is our target variable, "y." X represents our features, which is everything from the merged dataframe after we drop the following columns: "key","stars","helpful_yes","helpful_no","rating","sentiment."  These values add no value to sentiment, so we exclude them. 

```
# Segment the features from the target
y = df_tfidf_text["sentiment"]
X = df_tfidf_text.drop(["key","stars","helpful_yes","helpful_no","rating","sentiment"], axis=1)
```

After segmenting features from the target, we train, test and split the data at 75% with sklearn's train_test_split.

![train_test_split](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/train_test_split.png)


<br>
<br>

***9.  Balanced Random Forest Classifier***

For this project we decided to go with a Random Forest model (RF) for our classifier. There were several reasons we chose this over other models. First, the preprocessing required is compatible with an RF model. Also, a RF model is not as prone to overfitting the data as a Decision Tree and we thought that was a risk with this dataset. However, with a RF, we are limited with regression but that is not as important with this dataset which is another reason why we chose this model. Lastly, RF can be used to rank the importance of input variables in a natural way and can handle thousands of input variables without variable deletion.


The RF Model produces an Accuracy Score, Confusion Matrix and a Classification Report that highlights preciscion, recall and F1 scores. It also produces Top 10 Features Ranked by Importance.

![top_10_features_RF](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/top_10_features_RF.png)

We understand that neither the TF-IDF nor RF Models are completely trained model, but the model works and its ready for advance feature extraction and futher training. We still need to fine tune the features and reach an acceptable accuracy score.
