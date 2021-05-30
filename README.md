# UW_final_project
## Members: Brian Forth, Ashley Green, Max Thorstad, Kylie Wrenn

### Communication Protocols
For the duration of this project, Slack will be our primary communication vehicle. We've created a group chat there and so far it has proven sufficient to get feedback from one another and brainstorm.

We also have a standing weekly meeting at 7pm CST every Wednesday via Zoom, in addition to utilizing weekly class time to strategize next steps and perform our analysis.

### Presentation
We selected this topic to determine what ice cream flavor would be the best for our new ice cream company. For our new flavor, we want a specialty flavor. Our source data is several csv files from Kaggle that compiled flavors and customer reviews from four different ice cream companies. The flavor dataset includes the company, name of the flavor, and a description. The review dataset includes an overall product rating from 1-5, votes on whether or not other customers found each review helpful, and a written-out review that includes an individual consumer star rating of their product experience ranging from 1-5. 

We will use Natural Language Processing (NLP) to parse out the written reviews in order to determine which words are most associated with higher ratings so we can better determine a flavor for our specialty ice cream. This process will help us finalize our decision on a new specialty flavor and maximize efficiency in our product development process.


## Machine Learning Model
### Natural Language Processing (NLP)

NLP is the process of converting normal language to a machine readable format, which allows a computer to analyze text as if it were numerical data.  

#### Information Extraction & Text Classification
source: nlp_feature_extraction_vectorizing.ipynb

The primary goal of the nlp_feature_extraction_vectorizing.ipynb notebook is to pre-process large amounts of text data in order to prepare it for an NLP model that extract information and classifies text. 

- Extracting information: Many NLP tasks require the ability to retrieve specific pieces of information from a given document. We want to extract certain aspects of the review text data to present a compelling case for our speciality flavor.

- Classifying text: For many of the aforementioned use cases to work, a computer must know how to classify a given piece of text. We will classify our "Bag of Words" by positive and negative sentiment. 


#### Scope of the notebook:

##### 1.  Data Inspection
##### 2.  Add Sentiment Feature to data set
##### 3.  Create Product Sentiment Reviews Dataset
##### 4.  Tokenize "text" words
##### 5.  Bag of Words - Extract the most common words
##### 6.  Create Tokenized Reviews data set
##### 7.  TFID Vectorizing for Supervised  ML algorithms


##### 1.  Data Inspection

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

##### 2.  Add Sentiment Feature to data set

Here we assign a value of 1 to reflect positive sentiment. This consists of star rating greater than or equal to 4. Any review with a star rating less than 4 gets a value of 0 to reflect negative sentiment. Remember, star rating is the rating left by the individual reviewer. It is different than the overall rating presented by Amazon.

![add_sentiment](https://github.com/MaxThors/UW_final_project/blob/ash_seg2/Resources/Images/add_sentiment.png)