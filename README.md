# Detecting Political Bias in News Articles Using Machine Learning

## Abstract
Political bias in the media is becoming increasingly prevalent. Identifying and addressing slanted reporting is critical for transparency and reducing misinformation. This study focuses on detecting political biases by distinguishing between left-leaning and right-leaning news articles using machine learning techniques. We use a labeled dataset of U.S news articles from Kaggle, featuring text snippets and metadata. After cleaning the text and applying TF-IDF, we train two models: Logistic Regression as a baseline and Random Forest for a more complex, ensemble-based comparison. This allows us to assess how effectively machine learning can detect political bias in news content. 
	
## Introduction 
In today’s information age, media and news reporting bias plays a powerful role in shaping public discourse and influencing public opinion. As our modern political climate becomes increasingly polarized, the line between objective journalism and partisan reporting continues to blur. Political bias has become pervasive in the media, often leading to the distortion of facts, selective coverage, and the amplification of radicalized viewpoints. This growing divide in media representation not only fuels misinformation but also deepens ideological echo chambers, making it difficult for readers to form well-rounded, fact-based perspectives on current events.
Given the stakes, it is crucial that consumers of news are equipped with tools to critically assess the reliability and neutrality of their sources. One important step toward this goal is identifying whether a news article carries a political bias, specifically, whether it leans left or right on the ideological spectrum. Our project seeks to address this challenge by applying deep learning techniques to automatically detect and potentially even filter political bias in news articles based on their textual content and associated metadata.
By analyzing linguistic features such as word choice, tone, and thematic focus, as well as contextual metadata like publication source and article tags, we aim to build a predictive model that can classify articles as left-leaning or right-leaning. This approach not only has the potential to promote greater media transparency but also empowers readers to engage with news more critically, fostering a more informed and discerning public. Ultimately, our work contributes to ongoing efforts to mitigate the influence of bias in media and uphold the integrity of democratic discourse.
	
## Dataset
We used a dataset on Kaggle composed of scraped articles and metadata from various U.S.-based news outlets that had the following rows:
Article Text Snippet: A headline or the first sentence of the article
URL: A link to the full article
Source: The media outlet that published the piece (e.g., USA Today)
Topic: Article’s general Category
Bias Label: Categorical Indicator of bias - left, right, or center
Bias Score: A Subjective measure describing tone or perceived objectivity
For binary classification, only articles labeled as left or right would be considered; as such, we would need to process the data to remove any instances that are not marked as such (Entries marked center are excluded).

## Preprocessing
To prepare the dataset for analysis, the following preprocessing methods will be used:
Filtering: Removed rows with null or duplicate values and those labeled center
Text Cleaning: Stripped stop words, punctuation, converted text to lowercase, and removed all words less than 3 characters long
Using a library such as NLTK to remove the stop words
This is done to remove anything that might distract the model from learning functional patterns
Vectorization: Applied TF-IDF to extract important terms and down-weight common ones.
Label Encoding: Assigning binary values (0 for left, 1 for right)
Train/Test Split: 80/20 split to preserve label balance
80% of the data will be used to train, and 20% of the data will be used to test the model
Approaches: Logistic Regression, Random Forest Classification and RNNs
For the purposes of detecting political bias, we utilized three different kinds of machine learning models. These are Logistic Regression, Random Forest Classification, and Recurrent Neural Networks (RNNs). The former two of these models use TF-IDF vectorized text as input, allowing us to compare their performance using the same features. Meanwhile, the RNN model utilizes an LSTM layer with 64 instances and has a dropout of 0.3 for regularization and a dense sigmoid output layer. We tokenized the text and padded sequences to ensure uniform input length. For the purposes of the model, 0 represents a left-leaning bias while 1 represents a right-leaning bias.
	
## Results and Evaluation
### Format  
Model  
Train Accuracy  
Test Accuracy  
F1 Score (avg)  
##
Logistic Regression  
83%  
78%  
0.77  

Random Forest  
99.9% (overfit)  
60%  
0.61  
  
Random Forest (Tuned)  
71.9%  
62%  
0.60  

RNN  
86%  
86%  
0.86  


The performance of the models highlights important trade-offs between simplicity, interpretability, and predictive power. Logistic regression, while limited in capturing complex patterns, performed consistently well and provided effective at distinguishing between ideological word choices. It also served as a benchmark, helping to contextualize the performance of more complex models. Random Forest initially overfitted the training data, achieving nearly perfect training accuracy but struggling on the test set. After tuning, the model generalized better, though its performance remained slightly below that of logic regression.

The RNN model emerged as the most effective at capturing the nuances of political language, achieving high accuracy and F1 scores on both training and test data. Its ability to account for word order and context likely contributed to its superior performance. However, the cost of this performance was significantly increased training time and reduced interpretability. These results suggest that help learning models can offer valuable improvements in text classification tasks, but may be best suited for projects with the computational resources and need for high accuracy. Initially, RNN showed the model overfitting quickly, predicting only one class. After incorporating class weights and increasing the number of epochs from 5 to 20, the model achieved ~86% accuracy on the validation set. However, the model required significantly more computation and was harder to interpret. Despite this, it performed well in detecting subtle ideological framing missed by the other models.

Comparing the results across all models reveals meaningful insights into the strengths and limitations of each approach. Logistic Regression offers a reliable and computationally inexpensive baseline, particularly suitable when interpretability is crucial. Random Forest, while more powerful in theory, must be carefully tuned to avoid overfitting, which can mask its potential effectiveness. Its superior generalization performance demonstrates the benefits of sequence modeling text-based ideological detection, especially for tasks where subtle linguistic variations matter.. 

## Conclusion
Our project demonstrates that machine learning can effectively detect bias in news articles using only short text excerpts and metadata. While simple models like Logistic Regression provide decent baselines, more sophisticated approaches like Logistic Regression, provide decent baselines, more sophisticated approaches, like Random Forest and RNN’s offer improved performance and deeper insight into linguistic patterns. Notably, the RNN model showed the best test accuracy when appropriately regularized. 

Still, the ethical implications of automatic bias detection warrant caution. Mislabeling or over-reliance on automated systems could exacerbate polarization. Future work should focus on multi-class classification (including center), explainable AI methods for interpreting model decisions, and using full articles instead of snippets for richer context.

## References  
Spinde, Timo. “MBIC – a Media Bias Annotation Dataset.” Kaggle, 22 Jan. 2024, www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset?resource=download.  
Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 2011  
NLTK, www.nltk.org/. Accessed 11 May 2025.  
