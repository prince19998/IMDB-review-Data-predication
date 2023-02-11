#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('IMDB Dataset.csv')

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df['sentiment']=[1 if sentiment== 'positive' else 0 for sentiment in df['sentiment']]


# In[6]:


df.head()


# In[7]:


sns.countplot(df['sentiment'])
plt.show()


# In[8]:


df['sentiment'].value_counts()


# In[9]:


import nltk


# In[15]:


#nltk.download('stopwords')


# In[16]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()


# In[17]:


import re


# In[18]:


from tqdm import tqdm


# In[19]:


corpus = []
for i in tqdm(range(0,len(df))):
    sentence = re.sub('[^a-zA-Z]',' ',df['review'][i])#to read every sentence of the review column
    sentence = sentence.lower()
    sentence = sentence.split()
    sentence = [ps.stem(word) for word in sentence if not word in stopwords.words('english')]
    sentence = ' '.join(sentence)
    corpus.append(sentence)


# In[23]:


#corpus=[]
#for i in tqdm(range(0,len(df))):
#    sentence = re.sub('[^a-zA-Z]','',df['review'][i])  #to read every sentence of the review column
#    sentence=sentence.lower()
#    sentence=sentence.split()
#    sentence=[ps.stem(word) for word in sentence if not word in stopwords.words('english')]
#    sentence=''.join(sentence)
#    corpus.append(sentence)


# In[24]:


#corpus


# In[29]:


#bag of words model
from sklearn.feature_extraction.text import CountVectorizer #Bagofwords Model


# In[30]:


cv=CountVectorizer(max_features=2500)


# In[31]:


#independent varibale
X1=cv.fit_transform(corpus).toarray()


# In[32]:


y1=pd.get_dummies(df['sentiment'])
y1=y1.iloc[:,1].values


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train1, X_test1, y_train1, y_test1=train_test_split(X1,y1, test_size=0.2, random_state=1)


# In[35]:


X_train1.shape


# In[36]:


X_test1.shape


# In[37]:


y_train1.shape


# In[38]:


y_test1.shape


# In[39]:


from sklearn.naive_bayes import MultinomialNB


# In[40]:


model1=MultinomialNB()


# In[41]:


model1.fit(X_train1, y_train1)


# In[43]:


y_pred1=model1.predict(X_test1)


# In[44]:


from sklearn.metrics import accuracy_score,classification_report, confusion_matrix


# In[45]:


accuracy_score(y_test1, y_pred1)*100


# In[46]:


model1.score(X_train1,y_train1)*100


# In[47]:


print(classification_report(y_test1, y_pred1))


# In[49]:


cf1=confusion_matrix(y_test1,y_pred1)
cf1


# In[50]:


plt.figure(figsize=(10,5))
plt.title('confusion_Matrix_Bow', fontsize=20)
sns.heatmap(cf1,annot=True,fmt='g',cmap='Blues')
plt.show()


# In[51]:


plt.figure(figsize=(10,5))
plt.title('Confusion_Matrix_BOW',fontsize=20)
sns.heatmap(cf1,annot=True,fmt='g',cmap='Blues')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




