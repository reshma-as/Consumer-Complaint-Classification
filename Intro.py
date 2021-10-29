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
