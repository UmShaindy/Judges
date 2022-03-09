#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('/Users/naama/Desktop/Judges.csv')
print (df)


# In[ ]:


df = df[['genderEN', 'nationalityEN', 'religionEN', 'ethnicityEN',
        'citySocio', 'educationTypeEN', 'mainLegalEducationEN', 'mainLegalEducationTypeInstEN', 'typeAdvancedLegalEducationEN',
         'placeOfInternshipEN', 'lastPositionBeforeJudgeshipEN', 'position1EN', 
        'everInDistrict', 'everInSupremeCourt']]
print(df)


# In[ ]:


for string in df:
    df[string]=df[string].astype('category').cat.codes
df.corr()


# In[ ]:


plt.figure(figsize=(10,8))
plt.title('Correlation of Attributes with Class variable')
a = sns.heatmap(df.corr(), square=True, annot=True, fmt='.2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()


# In[ ]:


X = df.drop(['everInSupremeCourt'], axis=1)
y = df['everInSupremeCourt']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


X_train.shape, X_test.shape


# In[ ]:


X_train.dtypes


# In[ ]:


X_train.isnull().sum()


# In[ ]:


X_train.head()


# In[ ]:


cols = X_train.columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

X_train.head()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=3)


knn.fit(X_train, y_train)


# In[ ]:


y_pred = knn.predict(X_test)

y_pred


# In[ ]:


y_pred = knn.predict(X_test)

y_pred


# In[ ]:


knn.predict_proba(X_test)[:,0]


# In[ ]:


knn.predict_proba(X_test)[:,1]


# In[ ]:


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:


y_pred_train = knn.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[ ]:


knn_6 = KNeighborsClassifier(n_neighbors=6)
knn_6.fit(X_train, y_train)
y_pred_6 = knn_6.predict(X_test)

print('Model accuracy score with k=6 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_6)))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm_7 = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm_7)

print('\nTrue Positives(TP) = ', cm_7[0,0])

print('\nTrue Negatives(TN) = ', cm_7[1,1])

print('\nFalse Positives(FP) = ', cm_7[0,1])

print('\nFalse Negatives(FN) = ', cm_7[1,0])


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[ ]:


TP = cm_7[0,0]
TN = cm_7[1,1]
FP = cm_7[0,1]
FN = cm_7[1,0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# In[ ]:


classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# In[ ]:


precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# In[ ]:


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# In[ ]:


true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


# In[ ]:


false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


# In[ ]:


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# In[ ]:


y_pred_prob = knn.predict_proba(X_test)[0:10]

y_pred_prob


# In[ ]:


plt.figure(figsize=(6,4))

# adjust the font size 
plt.rcParams['font.size'] = 12


plt.hist(y_pred, bins = 10)


plt.title('Histogram of @@@')
plt.xlim(0,1)

plt.xlabel('Predicted probabilities of @@')
plt.ylabel('Frequency')


# In[ ]:




