# from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import warnings
warnings.filterwarnings('ignore')

x = pd.read_csv('creditcard.csv')
# x2 = pd.read_csv('binary/combined1.csv')
# print("x: ", x.head())
# print("x2: ", x2.head())
x = x[~x.isin([np.nan, np.inf, -np.inf]).any(1)]

df1 = x.copy()
df1_y = df1['Class']

df1 = x.loc[:, x.columns != 'Class']
# df1 = df1[~df1.isin([np.nan, np.inf, -np.inf]).any(1)]


# def normalize(df):
#     result = df.copy()
#     for feature_name in df.columns:
#         max_value = df[feature_name].max()
#         min_value = df[feature_name].min()
#         mean_value = df[feature_name].mean()
#         result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
#     return result


# df1 = normalize(df1)
df1 = df1.join(df1_y)
x = df1.copy()

df = pd.DataFrame(x, columns = x.columns)
# df['marker'] = x.target
X = df.drop('Class', 1)
y = df['Class']
len_features = X.shape[1]

# Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
plt.show()

# Correlation with output variable
cor_target = abs(cor['Class'])

# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.02]
print(relevant_features)
print(len(relevant_features))

# Backward Elimination
cols = list(X.columns)
pmax = 1
while len(cols) > 0:
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if pmax > 0.05:
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
print(len(selected_features_BE))


# RFE (Recursive Feature Elimination)

model = LogisticRegression()

# Initializing RFE model
rfe = RFE(model, 30)

# Transforming data using RFE
X_rfe = rfe.fit_transform(X, y)

# Fitting the data to model
model.fit(X_rfe, y)
print(rfe.support_)
print(rfe.ranking_)

# no of features
nof_list = np.arange(1, len_features)
high_score = 0
# Variable to store the optimum features
nof=0
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LogisticRegression('l2')
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if score > high_score:
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
