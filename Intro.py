#import statements

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#dataset display

df = pd.read_csv("consumer_Complaints.csv")
df.head()

#dispaly the number of rows and columns

df.shape

#display the datypes

df.dtypes

#statistical details

df.describe(include='all')

#Percentage of missing values in each column

df.isnull().sum()/df.shape[0]*100

#Exploratory Data Analysis

fig,ax = plt.subplots(figsize=(18,6))
sns.countplot(x='Product',data=df)

df.groupby('Product').Consumer_complaint.count().plot.bar(ylim=0)


#plot to understand which product has more customer disputes on their complaints after resolving the issues

pd.crosstab(df['Product'],df['Consumer disputed?']).plot(kind='bar')

#count of timely response

sns.countplot(x='Timely response?',data=df)

#text preprocessing steps

#uppercase to lowercase
df1['Consumer Complaint'] =df1['Consumer Complaint'].apply(lambda x: ' '.join([i.lower() for i in x.split()]))
df1['Consumer Complaint'].sample(2)


#removing punctuations
df1['Consumer Complaint'] =df1['Consumer Complaint'].str.replace(r'[^\w\s]',"")
df1['Consumer Complaint'].sample(2)

#removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
df1['Consumer Complaint'] =df1['Consumer Complaint'].apply(lambda x: ' '.join([i for i in x.split() if i not in stop]))from nltk.corpus import stopwords
df1['Consumer Complaint'].head(1)


#text standardization


dico = {}
dico1 = open('doc1.txt', 'rb')
for word in dico1:
    word = word.decode('utf8')
    word = word.split()
    dico[word[1]] = word[3]
dico1.close()
dico2 = open('doc2.txt', 'rb')
for word in dico2:
    word = word.decode('utf8')
    word = word.split()
    dico[word[0]] = word[1]
dico2.close()
dico3 = open('doc3.txt', 'rb')
for word in dico3:
    word = word.decode('utf8')
    word = word.split()
    dico[word[0]] = word[1]
dico3.close()
def txt_std(words):
    list_words = words.split()
    for i in range(len(list_words)):
        if list_words[i] in dico.keys():
            list_words[i] = dico[list_words[i]]
    return ' '.join(list_words)
df1['consumer_complaint_narrative'] = df1['consumer_complaint_narrative'].apply(txt_std)
df1.consumer_complaint_narrative.head(1)

#model building
#decision tree classifier


from sklearn.tree import DecisionTreeClassifier
text_clf=Pipeline([('tf',TfidfVectorizer(sublinear_tf= True, 
                       min_df = 5, 
                       norm= 'l2', 
                       ngram_range= (1,2), 
                       stop_words ='english') ),
                 ('clf',DecisionTreeClassifier())])
text_clf.fit(X_train, y_train)
text_clf.predict(['I have outdated information on my credit report'])[0]
y_pred=text_clf.predict(X_test)
print(metrics.classification_report(y_test,y_pred))
cv_results = cross_val_score(text_clf, 
                                 X_train, y_train, 
                                 cv=5,
                                 scoring="accuracy",
                                 n_jobs=-1)
print(np.mean(cv_results))





