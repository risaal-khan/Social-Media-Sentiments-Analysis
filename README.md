# Social Media Sentiments Analysis

# About
* The Social Media Sentiments Analysis Dataset captures a vibrant tapestry of emotions, trends, and interactions across various social media platforms. 
* This dataset provides a snapshot of user-generated content, encompassing text, timestamps, hashtags, countries, likes, and retweets. 
* Each entry unveils unique stories—moments of surprise, excitement, admiration, thrill, contentment, and more—shared by individuals worldwide.

![image](https://github.com/user-attachments/assets/87921f3a-ef25-4c55-8b38-a527d22ecbad)

## Key Features

| Feature    | Description                                           |
|------------|-------------------------------------------------------|
| **Text**   | User-generated content showcasing sentiments          |
| **Sentiment** | Categorized emotions                                 |
| **Timestamp** | Date and time information                            |
| **User**   | Unique identifiers of users contributing               |
| **Platform** | Social media platform where the content originated   |
| **Hashtags** | Identifies trending topics and themes                |
| **Likes**  | Quantifies user engagement (likes)                     |
| **Retweets** | Reflects content popularity (retweets)               |
| **Country** | Geographical origin of each post                      |
| **Year**   | Year of the post                                       |
| **Month**  | Month of the post                                      |
| **Day**    | Day of the post                                        |
| **Hour**   | Hour of the post                                       |

# Text Preprocessing
* Convert all text to lowercase to maintain consistency.
* Remove URLs and Links: Social media content often includes links, which do not add value to sentiment analysis.
* Remove Special Characters and Punctuation: Eliminate unnecessary characters (e.g., #, @, !) and symbols.
* Tokenization: Break the text into individual words.
* Stopwords Removal: Remove common stopwords (e.g., "the", "is", "a") that do not carry significant meaning.
* Lemmatization: Reduce words to their root form (e.g., "running" → "run").

           +-----------------------------+
           |  Convert all text to        |
           |  lowercase to maintain      |
           |  consistency                |
           +-----------------------------+
                            |
                            v
           +-----------------------------+
           |  Remove URLs and Links      |
           |  (social media content      |
           |  often includes links)      |
           +-----------------------------+
                            |
                            v
           +-----------------------------+
           |  Remove Special Characters  |
           |  and Punctuation            |
           |  (e.g., #, @, !)            |
           +-----------------------------+
                            |
                            v
           +-----------------------------+
           |  Tokenization               |
           |  (break the text into       |
           |  individual words)          |
           +-----------------------------+
                            |
                            v
           +-----------------------------+
           |  Stopwords Removal          |
           |  (remove common stopwords)  |
           +-----------------------------+
                            |
                            v
           +-----------------------------+
           |  Lemmatization              |
           |  (reduce words to their     |
           |  root form, e.g., "running" |
           |  → "run")                   |
           +-----------------------------+



# Sentiment Analysis
## Lexicon-Based Sentiment Analysis
* Lexicon-based methods assign a sentiment score to words using predefined libraries.
* Lexicon-based sentiment analysis is a technique used in natural language processing to detect the sentiment of a piece of text. It uses lists of words and phrases (lexicons or dictionaries) that are linked to different emotions to label the words (e.g. positive, negative, or neutral) and detect sentiment

![image](https://github.com/user-attachments/assets/f81c457a-5c3b-4e30-9e7f-551c508ebde5)

### Using VADER
* VADER stands for Valence Aware Dictionary and Sentiment Reasoner. 
* It’s a tool used for sentiment analysis, which is basically a way to figure out if a piece of text is expressing positive, negative, or neutral emotions.
* VADER doesn’t just say positive or negative, it assigns a score between -1 (most negative) and +1 (most positive) to show how strong the sentiment is.
###
* Importing SentimentIntensityAnalyzer from NLTK for sentiment analysis and downloading the vader_lexicon.
* Creating an instance of SentimentIntensityAnalyzer.
* Applying the sentiment analyzer to the 'Text' column of the dataframe df to calculate the compound sentiment score, storing the results in a new column Sentiment_Score.
* Defining a function classify_sentiment to classify the sentiment score into 'Positive', 'Negative', or 'Neutral' based on specific thresholds.
* Applying the classify_sentiment function to the Sentiment_Score column to create a new column Predicted_Sentiment

![image](https://github.com/user-attachments/assets/eb567a4d-59a2-4feb-9ee0-c938d35a5bb7)

![image](https://github.com/user-attachments/assets/df31d8c7-273d-49e3-9e20-f1613a1ce646)

### Using TextBlob

![image](https://github.com/user-attachments/assets/87f1b0b0-057b-4870-9c91-572740987f6c)

* TextBlob returns polarity and subjectivity of a sentence. 
* Polarity lies between [-1,1], -1 defines a negative sentiment and 1 defines a positive sentiment.
* Subjectivity lies between [0,1]. Subjectivity quantifies the amount of personal opinion and factual information contained in the text. The higher subjectivity means that the text contains personal opinion rather than factual information.
###
* Importing TextBlob for text processing and sentiment analysis.
* Applying TextBlob to the 'Text' column of the dataframe df to calculate the sentiment polarity, storing the results in a new column Sentiment_Polarity.
* Classifying Sentiment: Use a lambda function to classify the sentiment polarity into 'Positive', 'Negative', or 'Neutral' based on specific thresholds and store the results in a new column Predicted_Sentiment1

![image](https://github.com/user-attachments/assets/352e20bf-fe9c-4267-bdf1-2957307cf049)

![image](https://github.com/user-attachments/assets/e828a090-d8e5-460a-8a6b-a31283b1d43e)

## Machine Learning-Based Sentiment Analysis
Building a text classification model to predict sentiments.
####
* Label Encoding: Converting sentiment categories (e.g., "positive", "negative") into numerical values for model training.
* Feature Extraction: Converting text into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency).

### Model Training
* Split the data into training and testing sets and train a classification model (e.g., Logistic Regression)
* After training & testing got accuracy score as 11.56% which is very low
* A low accuracy score of 0.115 indicates that model may not be performing well due to various reasons like imbalanced data, poor feature representation, or incorrect assumptions about the problem.
* to overcome this we will use Resampling Techniques

### Resampling Techniques
* RandomOverSampler is a resampling technique used to handle imbalanced datasets, particularly in classification problems where one class has significantly fewer samples than the others.
* Creating an instance of RandomOverSampler and apply it to the feature set X and target y to balance the classes by oversampling the minority class.
* After training & Testing got 98.9% as accuracy score. But high accuracy score can be because of overfitting.
* to confirm this i will use cross validation & gridsearchcv.
* After Applying cross validation got this [0.99063518 0.9910387  0.99063136 0.99063136 0.99144603]
* After Applying gridsearchcv got best parameters as {'C': 10, 'solver': 'liblinear'} and score as 99.23%.

### Trying another models
* Random forest: accuracy score as 99.1%
* XGBoost: 98.8%

## Temporal Analysis
#### Sentiment Trends Over Months
-> Line Trend
* Starts very high in January (~170+), then sharply drops to February (~130).
* After that, there is a fluctuating trend with alternating increases and decreases.
* Sentiments show gradual recovery towards the end of the year (November–December).

-> Shaded Area
* Indicates variability or uncertainty in the data (likely confidence intervals).
* Higher variability occurs at the beginning of the year and mid-year.
* Months like February and July have narrower confidence intervals, indicating more stable sentiment trends.

![image](https://github.com/user-attachments/assets/98559dd7-a81e-4118-abf3-47218eab9fa2)

#### Sentiment Trends Over Years
-> Line Trend
* Starts low in 2010 (~70 sentiment), spikes sharply in 2011 (~225), then drops in 2014.
* From 2014 onward, the sentiment gradually stabilizes and fluctuates within a smaller range.
* A noticeable dip occurs after 2020 and recovers by 2023.

-> Shaded Area
* High variability is seen in 2011, indicating a lot of uncertainty during that year.
* The shaded area reduces significantly in later years (2016–2023), suggesting more stable sentiment trends.
* Sentiments show longer-term trends with a massive spike in 2011 and stabilization afterward. This could suggest a specific event or trend in 2011 that significantly impacted sentiment.

![image](https://github.com/user-attachments/assets/f881a316-7b53-4e37-9964-f1d73d58e556)

## User Behavior Insights
to identify which sentiments get the most engagement

![image](https://github.com/user-attachments/assets/7f7246bf-44e8-4073-b65c-92acbeed9b0d)

## Platform-Specific Analysis
Comparing how sentiments vary across platforms

![image](https://github.com/user-attachments/assets/54015ef3-3ed7-49b2-af54-b03fac01399f)

* Instagram shows a higher engagement in positive emotions such as Joy and Excitement, indicating a more cheerful user base.
* Facebook has a balanced mix of Neutral and positive sentiments, suggesting a diverse emotional expression.
* Twitter displays a wider variety of sentiments, with substantial expressions of Curiosity and Sadness, potentially indicating a platform where users engage in more varied and intense discussions.

## Hashtag Trends
Top Hashtag


![image](https://github.com/user-attachments/assets/fede0e68-386c-4cc4-9d1e-1df03d50527b)

## Geographical Analysis

![image](https://github.com/user-attachments/assets/4e0197e4-1dcf-4052-beaf-ef6a1a1f278e)

* Brazil, Germany, Netherlands and India, show a higher engagement in positive emotions such as Joy, Excitement, and Positive sentiments, indicating a generally optimistic and cheerful user base.
* USA, Canada, UK and Australia have balanced distributions with significant Happy and Contentment sentiments, indicating a positive outlook among users.

## User Identification

![image](https://github.com/user-attachments/assets/fc88b9bd-debc-4bd0-8c01-877a9ee4a6cd)

## Cross-Feature Analysis

![image](https://github.com/user-attachments/assets/4a6c2e43-b3b6-4d11-bd4b-66f8c0ad3d4c)

![image](https://github.com/user-attachments/assets/809aea77-e748-443a-b9e4-3ddc9c7163e8)

-> Twitter

* Generally, Twitter shows a wide range of sentiment scores with noticeable outliers in some countries.
* For example, the USA and UK have broad distributions with significant positive and negative sentiments.

-> Instagram
* Instagram has a more consistent distribution of sentiment scores across countries compared to Twitter.
* Positive sentiments are more concentrated, and there are fewer extreme outliers.

-> Facebook
* Facebook shows varied sentiment distributions similar to Twitter but tends to have a higher median sentiment score in several countries.
* Countries like India and Brazil exhibit higher positive sentiments on Facebook.

I had created a streamlit app which is deploy on streamlit cloud and below is the link to the app
https://social-media-sentiments-analysis-emzyvnl765y3zumdcqtckd.streamlit.app/
