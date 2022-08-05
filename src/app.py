import pandas as pd
import numpy as np
import pickle
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')
def caracteres_no_alfanumericos(text):
    """
    Sustituye caracteres raros, no digitos y letras
    Ej. hola 'pepito' como le va? -> hola pepito como le va
    """
    return re.sub("(\\W)+"," ",text)
def esp_multiple(text):
    """
    Sustituye los espacios dobles entre palabras
    """
    return re.sub(' +', ' ',text)
def url(text):
    return re.sub(r'(https://www|https://)', '', text)
df = df_raw.copy()
df = df.drop_duplicates().reset_index(drop = True)
df['url_limpia'] = df['url'].apply(url).apply(caracteres_no_alfanumericos).apply(esp_multiple)
df['is_spam'] = df['is_spam'].apply(lambda x: 1 if x == True else 0)
df.to_csv('/workspace/NLP-Project/data/processed/df.csv', index = False, encoding='utf-8')
vec = CountVectorizer().fit_transform(df['url_limpia'])
X_train, X_test, y_train, y_test = train_test_split(vec, df['is_spam'], stratify = df['is_spam'], random_state = 15)
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(random_state=15),param_grid,verbose=2)
grid.fit(X_train,y_train)
best_model = grid
filename = '/workspace/NLP-Project/models/best_model.pickle'
pickle.dump(best_model, open(filename,'wb'))
