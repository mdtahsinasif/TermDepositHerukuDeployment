# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 01:37:58 2020

@author: TahsinAsif
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:11:35 2020

@author: TahsinAsif
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import os
import joblib
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree,neighbors, model_selection,svm, ensemble,feature_selection,naive_bayes,linear_model



terms_df = pd.read_csv("C:/Users/TahsinAsif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/21112019/AI/Yogen/bank-full.csv",encoding='ISO-8859-1')
print(terms_df)

#Converting final out yes for 1 and no for 0

terms_df['y'] = terms_df.y.map({ 'yes': 1,
       'no': 0,
       })

terms_df['default'] = terms_df.default.map({ 'yes': 1,
       'no': 0,
       })

terms_df['housing'] = terms_df.housing.map({ 'yes': 1,
       'no': 0,
       })
terms_df['loan'] = terms_df.loan.map({ 'yes': 1,
       'no': 0,
       })


print(terms_df.head())


print(terms_df.shape)
print(terms_df.info())

#categorical columns: numerical EDA

pd.crosstab(index=terms_df["job"], columns="count")
pd.crosstab(index=terms_df["marital"], columns="count")  
pd.crosstab(index=terms_df["education"],  columns="count")
pd.crosstab(index=terms_df["contact"],  columns="count")

#categorical columns: visual EDA
sns.countplot(x='job',data=terms_df)
sns.countplot(x='marital',data=terms_df)
sns.countplot(x='education',data=terms_df)
sns.countplot(x='contact',data=terms_df)

#continuous features: visual EDA
sns.boxplot(x='age',data=terms_df)
sns.distplot(terms_df['age'])
sns.distplot(terms_df['age'], kde=False)
sns.distplot(terms_df['age'], bins=20, rug=True, kde=False)
sns.distplot(terms_df['age'], bins=100, kde=False)

#continuous features: numerical EDA
terms_df['age'].describe()

#balance eda
#continuous features: visual EDA
sns.boxplot(x='balance',data=terms_df)
sns.distplot(terms_df['balance'])
sns.distplot(terms_df['balance'], kde=False)
sns.distplot(terms_df['balance'], bins=20, rug=True, kde=False)
sns.distplot(terms_df['balance'], bins=100, kde=False)

#continuous features: numerical EDA
terms_df['balance'].describe()


terms_df.describe()

#explore bivariate relationships: categorical vs categorical 
sns.factorplot(x="job", hue="y", data=terms_df, kind="count", size=6)
sns.factorplot(x="marital", hue="y", data=terms_df, kind="count", size=6)
sns.factorplot(x="education", hue="y", data=terms_df, kind="count", size=6)
sns.factorplot(x="contact", hue="y", data=terms_df, kind="count", size=6)
#explore bivariate relationships: continuous  vs categorical
sns.FacetGrid(terms_df, hue="y",size=8).map(sns.kdeplot, "balance").add_legend()
sns.FacetGrid(terms_df, row="y",size=8).map(sns.distplot, "age").add_legend()
sns.FacetGrid(terms_df, hue="y",size=8).map(sns.distplot, "campaign").add_legend()
sns.FacetGrid(terms_df, hue="y",size=8).map(sns.distplot, "duration").add_legend()
#explore bivariate relationships: continuous vs continuous 
sns.jointplot(x="balance", y="age", data=terms_df)

#Model Creation 
#converting categorical data
terms_df = pd.get_dummies(terms_df, columns=['job','marital','education',
                                             'contact','poutcome'])
terms_df.head()
#train - test split
features = ['job_admin','job_blue-collar','job_entrepreneur','job_housemaid','job_management','job_retired','job_self-employed','job_services','job_student','job_technician','job_unemployed','job_unknown',
            'marital_divorced','marital_married','marital_single',
            'education_primary','education_secondary','education_tertiary','education_unknown',
            'contact_cellular','contact_telephone','contact_unknown',
            'age',
            'loan',
            'housing',
            'default','balance','campaign','duration','day','pdays','previous',
            'poutcome_failure','poutcome_other','poutcome_success','poutcome_unknown']
X_train, X_test, Y_train, Y_test = train_test_split(terms_df[features], terms_df['y'], random_state = 0)
print(X_train.head())

classifier = ensemble.BaggingClassifier(random_state=100, n_jobs=1, verbose=1)
bt_grid = {'n_estimators':[5,10,20,30,40], 'max_features':[3,4,5,11]}
grid_classifierVersio1 = model_selection.GridSearchCV(classifier, bt_grid, cv=10, refit=True, return_train_score=True)
grid_classifierVersio1.fit(X_train, Y_train)
y_pred =  grid_classifierVersio1.predict(X_test)
y_pred1 =  grid_classifierVersio1.predict_proba(X_test)

from sklearn.metrics import accuracy_score,roc_auc_score
print("Accuracy Score:{}".format(accuracy_score(Y_test,y_pred)))
#Accuracy Score:0.8927718304874812
print("roc_auc_score Score:{}".format(roc_auc_score(Y_test,y_pred1[:,1])))
#roc_auc_score Score:0.8682355521770537

results = grid_classifierVersio1.cv_results_
print(results.get('params'))
print(results.get('mean_test_score'))
print(results.get('mean_train_score'))
print(grid_classifierVersio1.best_params_)
#{'max_features': 11, 'n_estimators': 10}
print(grid_classifierVersio1.best_score_)
#0.8953638078933563
final_model = grid_classifierVersio1.best_estimator_
final_model.estimators_




##### creating Naive Bays Model ########
nb_estimator = naive_bayes.GaussianNB()
nb_estimator.fit(X_train, Y_train)

print(nb_estimator.class_prior_)
print(nb_estimator.sigma_)
print(nb_estimator.theta_)

res = model_selection.cross_validate(nb_estimator, X_train, Y_train, cv=10)
print(res.get('test_score').mean())
print(nb_estimator.score(X_train, Y_train))
#0.832399433761944

y_pred =  nb_estimator.predict(X_test)
y_pred1 =  nb_estimator.predict_proba(X_test)

print("Accuracy Score:{}".format(accuracy_score(Y_test,y_pred)))
#Accuracy Score:0.8927718304874812
print("roc_auc_score Score:{}".format(roc_auc_score(Y_test,y_pred1[:,1])))
#roc_auc_score Score:0.7988427370403567



#########################################

###################################
#linear svm
lsvm_estimator = svm.LinearSVC(max_iter=50000,random_state=100)
lsvm_grid = {'C':[0.1,0.2,0.5,1] }
grid_lsvm_estimator = model_selection.GridSearchCV(lsvm_estimator, lsvm_grid, cv=10)
grid_lsvm_estimator.fit(X_train, Y_train)
print(grid_lsvm_estimator.best_params_)
final_estimator = grid_lsvm_estimator.best_estimator_
print(final_estimator.coef_)
print(final_estimator.intercept_)
print(grid_lsvm_estimator.best_score_)
print(final_estimator.score(X_train, Y_train))

##################################
#Knn Algorithm
knn_estimator = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors':list(range(2,20,1)), 'weights':['uniform', 'distance'] }
knn_grid_estimator = model_selection.GridSearchCV(knn_estimator, knn_grid, scoring='accuracy', cv=10, refit=True, return_train_score=True)
knn_grid_estimator.fit(X_train, Y_train)

print(knn_grid_estimator.best_score_)
print(knn_grid_estimator.best_params_)
final_estimator = knn_grid_estimator.best_estimator_
print(final_estimator.score(X_train, Y_train))
#print(final_estimator.feature_importances_)
#0.8878141340647506

y_pred =  final_estimator.predict(X_test)
y_pred1 =  final_estimator.predict_proba(X_test)

print("Accuracy Score:{}".format(accuracy_score(Y_test,y_pred)))
#Accuracy Score:0.8864903123064674
print("roc_auc_score Score:{}".format(roc_auc_score(Y_test,y_pred1[:,1])))
#roc_auc_score Score:0.8180978530124766
###################


path = 'C:/Users/TahsinAsif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/21112019/AI/Yogen/'
joblib.dump(final_model,os.path.join(path, 'grid_classifier_modelVersion1.pkl') )
joblib.dump(final_estimator,os.path.join(path, 'knn_estimator_model.pkl') )

# Best feature selection
best_k = feature_selection.SelectKBest(feature_selection.chi2, k=10)
best_k.fit(X_train, Y_train)
print(best_k.get_support())
print(best_k.scores_)
terms_df1 = terms_df
new_features = X_train.columns[best_k.get_support()]
X_train1 = X_train[new_features]

#build model on X_train1
dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_grid = {'criterion':['gini','entropy'], 'max_depth':[3,4,5,6,7,8]}
grid_dt_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv=10)
grid_dt_estimator.fit(X_train1, Y_train)

print(grid_dt_estimator.best_params_)
print(grid_dt_estimator.best_score_)
print(grid_dt_estimator.score(X_train1, Y_train))
#0.8830954347056742
#Model evaluation
results = model_selection.cross_validate(grid_classifier, X_train, Y_train, cv = 10)
print(results.get('test_score').mean())
#0.8829185097807922

grid_dt_estimator.predict()
