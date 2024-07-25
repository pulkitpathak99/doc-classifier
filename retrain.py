import math
from creme import compose
from creme import feature_extraction
from creme import naive_bayes
import creme
from creme import metrics
from sklearn.model_selection import train_test_split
import pickle
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords
import pandas as pd
import nltk

from helper import preprocesse_df
nltk.download('punkt')
nltk.download('stopwords')
stopwords.words('english')
string.punctuation
ps = PorterStemmer()

# Load the model from the file
with open('predict-doc.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

filename = "Legal Documents"

df = pd.read_csv('./db/train/'+filename+'.csv')

# only run this if it has empty cols:
del df['Unnamed: 2']
del df['Unnamed: 3']
del df['Legal Document']

# df.rename(columns={'Content': 'content',
#           'Category': 'category'}, inplace=True)

new_df = preprocesse_df(df)

transformed_df = new_df[['transformed_content', 'category']]

message_train, message_test = train_test_split(transformed_df)

messages_train = message_train.to_records(index=False)
messages_test = message_test.to_records(index=False)

model = loaded_model

metric = metrics.Accuracy()
# Training the model row by row
for content, category in messages_train:
    model = model.fit_one(content, category)
    y_pred = model.predict_one(content)
    metric = metric.update(category, y_pred)
    print(metric)

# test Data Accuracy
test_metric = metrics.Accuracy()
for content, category in messages_test:
    y_pred = model.predict_one(content)
    test_metric = metric.update(category, y_pred)
    print(test_metric)


with open('predict-doc.pkl', 'wb') as f:
    pickle.dump(model, f)
