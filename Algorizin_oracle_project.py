#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=Warning)
import pandas as pd
import numpy as np
import math as mt
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact


# # Loading Data set

# In[2]:


algdat=pd.read_csv('student-algorizin.csv',skiprows=1)


# In[3]:


algdat.head()


# # Data Cleaning

# ### Checking for null/ Missing values 

# In[4]:


algdat.shape


# In[5]:


algdat.info()


# In[6]:


algdat.isnull().value_counts()


# In[ ]:





# In[ ]:





# ### Change Grade to LE3 if grades<70 else GT3

# In[7]:


algdat['grades'] = algdat['grades'].apply(lambda x: 'bel70' if x == 'LE3' else 'abv70')


# ### Change Funding T to 'yes' and A to 'No'

# In[8]:


algdat['funding'] = algdat['funding'].apply(lambda x: 'yes' if x =='T' else 'no')


# In[9]:


algdat.head(10)


# ### For 'degree' change at_home to ‘Bachelors’, health, services and teachers to ‘Masters’, and others to ‘Doctorate’  

# In[10]:


def change_degree(degrees):
    if degrees=='at_home':
        return 'Bachelors'
    if degrees=='other':
        return 'Doctorate'
    else:
        return 'Masters'

algdat['degree'] = algdat['degree'].apply(change_degree)


# ### For‘country’ change teacher to ‘Bangladesh’, other to ‘Nepal’, health to ‘Nigeria’, services to ‘India’, teachers to ‘Pakistan’

# In[11]:


def change_country(var):
    if var=='teacher':
        return 'Bangladesh'
    if var=='other':
        return 'Nepal'
    if var=='health':
        return 'Nigeria'
    if var=='services':
        return 'India'
    else:
        return 'Pakistan'

algdat['country'] = algdat['country'].apply(change_country)


# In[12]:


algdat['reason'].unique()


# ### For ‘reason’ change course to ‘access’, home to ‘mentoring’, other to ‘immigration’ and reputation unchanged.

# In[13]:


def change_reason(var):
    if var=='course':
        return 'access'
    if var=='home':
        return 'mentoring'
    if var=='other':
        return 'immigration'
    else:
        return var

algdat['reason'] = algdat['reason'].apply(change_reason)


# ### For ‘finance’ change mother to ‘poor’ and father to ‘fine’ 

# In[14]:


algdat['finance'] = algdat['finance'].apply(lambda x: 'poor' if x =='mother' else 'fine')


# In[15]:


algdat.head()


# In[ ]:





# # Exploratory analysis

# Lets make a function to visualize the data

# In[16]:


def exploratory_plot(col_name, continuous):
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20,12), dpi=180)
    
    if continuous:
        sns.distplot(algdat.loc[algdat[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(algdat[col_name], order=sorted(algdat[col_name].unique()), saturation=1, ax=ax1)
    ax1.set_xlabel(col_name,fontsize = 20)
    ax1.set_ylabel('Count',fontsize = 20)
    ax1.set_title(col_name,fontsize = 20)

    if continuous:
        sns.boxplot(x=col_name, y='J', data=loans, ax=ax2)
        ax2.set_ylabel('',fontsize = 20)
        ax2.set_title(col_name + ' by Job Landing Success',fontsize = 20)
    else:
        ax2=sns.countplot(x='J', hue=col_name, data=algdat)
        ax2.set_ylabel('Count for each score',fontsize = 20)
        ax2.set_title('success by ' + col_name,fontsize = 20)
        ax2.set_xlabel('Job landing score',fontsize = 20)
    plt.legend(loc='upper right', title=col_name, fontsize = 20)
    plt.tight_layout()


# ### Plotting features  

# In[17]:


"""
for colname in list(algdat):
    if colname != 'A1' or colname != 'A2' or colname != 'J':
        exploratory_plot(colname,False)
"""


# In[18]:


algdat.describe()


# In[19]:


#algdat['J']=0


# In[20]:


for colname in list(algdat):
    print(algdat.groupby('J')[colname].describe())


# In[21]:


#algdat.loc[(algdat['J']==10) & (algdat['country'] =='Nepal') &(algdat['funding']=='no')]


# In[22]:


#algdat.loc[algdat['J'].isin([10,12])]


# In[23]:


algdat['reason'].unique()


# ### Separating the target variable (y) and make a dataframe of independent variables (X) 

# In[ ]:





# In[24]:


algdat.head(10)


# In[25]:


y=algdat.J


# In[26]:


y


# In[27]:


X=algdat.drop('J',axis='columns')


# In[28]:


X


# ## Separating train and test data from X, y 

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[31]:


X_train.shape, X_test.shape


# In[ ]:





# In[ ]:





# ## Feature Selection 

# ### F.1 Remove correlated depended feature

# If we find multiple correlated "independent features", then we can keep one, and drop others. There is no exact rule for the threshold score, but in general 0.85 is a good number. However, we can comeback later to improve.

# #### Using heatmap for correlation strength

# In[32]:


plt.figure(figsize=(12,10))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# In[33]:


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[34]:


corr_features = correlation(X_train, 0.8)
len(set(corr_features))


# In[35]:


print("The feature we can drop is ",corr_features)


# In[36]:


#X_train=X_train.drop(columns=['A1','A2'])


# In[ ]:





# ### F.2. Mutual information

# Some numerical features are strongly correlated with dependent/target variable. Lets find the most strong features and we will change the number of features later for performance improvement.

# 

# In[37]:


x_num=X_train.select_dtypes(exclude='object')


# In[38]:


from sklearn.feature_selection import mutual_info_classif
# determine the mutual information
mutual_info = mutual_info_classif(x_num, y_train)
mutual_info


# In[39]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = x_num.columns
mutual_info.sort_values(ascending=False)


# In[40]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))


# In[41]:


from sklearn.feature_selection import SelectKBest


# In[42]:


sel_k_cols = SelectKBest(mutual_info_classif, k=2)
sel_k_cols.fit(x_num, y_train)
X_list=x_num.columns[sel_k_cols.get_support()]


# In[43]:


X_list


# In[44]:


x_num.columns


# In[45]:


drop_list=x_num.drop(columns=X_list)


# In[46]:


drop_list.columns


# In[47]:


X_train=X_train.drop(columns=drop_list.columns)


# In[48]:


X_train.columns


# ### Chi square test 

# To find the correlation between categorical or discrete variables with dependent/target variable we can use Chi square test.

# In[49]:


X_train.astype('object')


# In[50]:


x_cat=X_train.select_dtypes(include='object')


# In[51]:


x_cat.shape


# In[52]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()


# In[53]:


for i in list(x_cat):
    x_cat[i]=label_encoder.fit_transform(x_cat[i])


# In[54]:


x_cat.head()


# In[55]:


from sklearn.feature_selection import chi2
f_p_values=chi2(x_cat,y_train)


# In[56]:


f_p_values


# In[57]:


p_values=pd.Series(f_p_values[1])
p_values.index=x_cat.columns
p_values.sort_index(ascending=True)


# In[58]:


X_new = SelectKBest(chi2, k=2)
X_new.fit(x_cat, y_train)

cols = X_new.get_support(indices=True)
features_df_new = x_cat.iloc[:,cols]


# In[59]:


features_df_new.columns


# In[60]:


drop_list2=x_cat.drop(columns=features_df_new.columns)


# In[61]:


drop_list2.columns


# In[62]:


X_train=X_train.drop(columns=drop_list2.columns)


# In[63]:


X_train.shape


# In[64]:


X_train


# ### Remove the same features from the test dataset 

# In[65]:


drop_list_test=list(set(X.columns)^set(X_train.columns))


# In[66]:


drop_list_test


# In[67]:


X_test=X_test.drop(columns=drop_list_test)


# In[68]:


X_test


# In[69]:


X_train.columns, X_test.columns


# # Feature Transformation

# Before this step, all features must be selected. Let's transform categorical features.

# ### Categorical feature transformation using one-hot-encoding!

# In[70]:


#algdat.info()


# In[71]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer, make_column_selector


# In[72]:


column_trans=make_column_transformer((OneHotEncoder(),make_column_selector(dtype_include=object)),(MinMaxScaler(),make_column_selector(dtype_include=np.number)))
#column_trans=make_column_transformer((OneHotEncoder(),make_column_selector(dtype_include=object)),remainder='passthrough')


# In[73]:


nptable_train=column_trans.fit_transform(X_train)
nptable_test=column_trans.fit_transform(X_test)


# In[74]:


#column_trans.get_feature_names()


# In[75]:


#newdf=pd.DataFrame(nptable)


# In[76]:


#newdf


# In[77]:


#nptable.dtype


# In[78]:


nptable_train, nptable_test


# In[ ]:





# In[79]:


#X=np.delete(nptable,-1,1)


# In[80]:


#y


# In[81]:


X_train.reason.unique()


# In[82]:


column_trans.transformers_


# In[83]:


#algdat.head()


# In[84]:


column_trans


# In[ ]:





# # Implementing ML algorithms

# In[85]:


from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.pipeline import make_pipeline


# To build Machine learning models we need to call distinct classifiers for different models. Then we need to build a pipeline, and we will measure the accuracy by cross validation testing. This will tell us how good these models will perform to the unseen data. Once the model is selected will do the hyperparameter tuning.

# ### Calling ML Classifiers

# In[86]:


linreg=LinearRegression()


# In[87]:


logreg=LogisticRegression(solver='liblinear',multi_class='ovr')


# In[88]:


rand_for=RandomForestClassifier(n_estimators=300)


# In[89]:


svc_model=SVC()


# In[90]:


decision_tree=DecisionTreeClassifier()


# In[91]:


Naive_bayes=GaussianNB()


# In[92]:


Knn= KNeighborsClassifier()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Building Pipelines 

# In[93]:


pipe_linreg=make_pipeline(column_trans,linreg)


# In[94]:


pipe_logreg=make_pipeline(column_trans,logreg)


# In[95]:


pipe_randforest=make_pipeline(column_trans,rand_for)


# In[96]:


pipe_svc=make_pipeline(column_trans,svc_model)


# In[97]:


pipe_dec_tree=make_pipeline(column_trans,decision_tree)


# In[98]:


pipe_naive=make_pipeline(column_trans, Naive_bayes)


# In[99]:


pipe_knn=make_pipeline(column_trans, Knn)


# In[100]:


pipe_linreg_web=make_pipeline(linreg)


# In[ ]:





# ### Cross validation score analysis 

# In[101]:


X_train_web=X_train.drop(columns=['reason','relocation'])
X_test_web=X_test.drop(columns=['reason','relocation'])


# In[ ]:





# In[102]:


cv_score_linear_regression=cross_val_score(pipe_linreg,X_train,y_train,cv=10,scoring='r2').mean()


# In[103]:


cv_score_logistic_regression=cross_val_score(pipe_logreg,X_train,y_train,cv=10,scoring='accuracy').mean()


# In[104]:


cv_score_random_forest=cross_val_score(pipe_randforest,X_train,y_train,cv=10,scoring='accuracy').mean()


# In[105]:


cv_score_SVM =cross_val_score(pipe_svc,X_train,y_train,cv=10,scoring='accuracy').mean()


# In[106]:


cv_score_decision_tree=cross_val_score(pipe_dec_tree,X_train,y_train,cv=10,scoring='accuracy').mean()


# In[107]:


cv_score_Naive_bayes=cross_val_score(pipe_naive,X_train,y_train,cv=10,scoring='accuracy').mean()


# In[108]:


cv_score_KNN=cross_val_score(pipe_knn,X_train,y_train,cv=10,scoring='accuracy').mean()


# In[109]:


cv_score_linear_regression_web=cross_val_score(pipe_linreg_web,X_train_web,y_train,cv=10,scoring='r2').mean()


# In[110]:


cv_score_linear_regression_web


# In[111]:


method_list=['Linear Regression','Logistic Regression','Random_Forest','Support Vector Machine','Decision Tree','Naive Bayes','KNN']

Score=[cv_score_linear_regression,cv_score_logistic_regression,cv_score_random_forest,cv_score_SVM,cv_score_decision_tree,cv_score_Naive_bayes,cv_score_KNN]


# In[112]:


Score


# In[113]:


score_table=pd.DataFrame(list(zip(method_list, Score)),columns =['Algorithm', 'Model accuracy score'])


# In[114]:


score_table


# In[115]:


plt.figure(figsize=(10,5))
plt.xlabel('Models',fontsize=15)
plt.ylabel ('Model accuracy scores',fontsize=15)
plt.title('Comparison of the algorithms',fontsize=15)

ax=sns.barplot(x=score_table['Algorithm'],y=score_table['Model accuracy score'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()


# ### Make some Prediction

# In[116]:


nptable_pd_train=pd.DataFrame(nptable_train)
pipe_linreg.fit(nptable_pd_train,y_train)


# In[117]:


linreg.coef_


# In[118]:


nptable_pd_test=pd.DataFrame(nptable_test)


# In[119]:


nptable_pd_test


# In[ ]:





# In[120]:


y_linreg_pred=pipe_linreg.predict(nptable_pd_test)


# In[121]:


y_linreg_pred


# In[122]:


r2_score(y_test,y_linreg_pred)


# In[123]:


plt.figure(figsize=(10,5))
plt.xlabel('Test samples',fontsize=15)
plt.ylabel ('Predicted outcome',fontsize=15)
plt.title('Comparison of test vs prediction',fontsize=15)

ax=sns.scatterplot(x=y_test,y=y_linreg_pred)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()


# In[ ]:





# In[125]:


pipe_linreg_web.fit(X_train_web,y_train)


# In[126]:


y_linreg_pred_web=pipe_linreg_web.predict(X_test_web)


# In[127]:


y_linreg_pred_web


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


pipe_randforest.fit(X_train,y_train)


# In[ ]:


y_pred = pipe_randforest.predict(X_test)


# In[ ]:





# In[ ]:


accuracy_score(y_test,y_pred, normalize=True)


# In[ ]:


sns.scatterplot(x=y_test,y=y_pred,)


# ## Hyperparameter tuning

# ### Linear Regression 

# In[ ]:


grid_lin_reg = GridSearchCV(LinearRegression(),param_grid = [{'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}], cv=10)
grid_lin_reg.fit(nptable_train, y_train)


# In[ ]:


grid_lin_reg.best_params_


# In[ ]:


lin_reg_df = pd.DataFrame(grid_lin_reg.cv_results_)


# In[ ]:


lin_reg_df


# In[ ]:


lin_reg_df[['param_copy_X','param_fit_intercept','param_normalize','mean_test_score']]


# ### Support Vector Machine

# In[ ]:



grid_svm = GridSearchCV(SVC(),param_grid = [{'C': [0.01,1, 10, 100, 1000], 'kernel': ['linear', 'rbf']}], cv=10)
grid_svm.fit(nptable_train, y_train)


# In[ ]:





# In[ ]:


grid_svm.best_params_


# In[ ]:


svm_df = pd.DataFrame(grid_svm.cv_results_)


# In[ ]:


svm_df


# In[ ]:


svm_df[['param_C','param_kernel','mean_test_score']]


# In[ ]:





# In[ ]:

file = open('algorizin.pkl', 'wb')
pickle.dump(pipe_linreg_web, file)

