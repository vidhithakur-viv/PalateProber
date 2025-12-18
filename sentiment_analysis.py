
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import re
#data = pd.read_excel(r"C:\ACER\ml_projects\sentiment_project\Reviews.xlsx")
#print(f"dataset shape{data.shape}")
data = pd.read_csv(r"C:\ml_projects\sentiment_project\Reviews.csv")

print(f"dataset shape {data.shape}")
data.head()
#Columns Features
print(f"Feature Name:{data.columns.values}")
#Check for null values 
data.isnull().sum()
