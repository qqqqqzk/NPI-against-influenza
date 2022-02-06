import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
kf = KFold(n_splits=10)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import shap

df = pd.read_csv('data.csv', )

X = f[['Lag_5_C1_sub_index', 'Lag_4_C2_sub_index', 'Lag_5_C3_sub_index',
       'Lag_3_C4_sub_index', 'C5_sub_index', 'Lag_1_C6_sub_index',
       'Lag_2_C7_sub_index', 'Lag_2_C8_sub_index', 'E1_sub_index', 'E2_sub_index', 
       'Lag_7_H1_sub_index', 'H2_sub_index', 'Lag_4_H3_sub_index', 'Lag_4_H6_sub_index']]
y = f[['dif']]

X.columns = ['School closure', 'Workplace closure', 'Public events cancellation', 
'Gathering limitation', 'Public transport suspension', 'Stay at home requirement', 
'Domestic travel restriction', 'International travel restriction', 
'Unemployment subsidy', 'Debt/contract relief', 'Health education promotion', 
'Testing policy', 'Contact tracing', 'Mask wearing requirement']

# XGBoost
xgb = XGBRegressor(reg_alpha=0.0001, colsample_bytree=1, gamma=0.29999999999999993, reg_lambda=0.991, 
                   learning_rate=0.3027546451247583, max_depth=6, min_child_weight=1, n_estimators=114, subsample=0.9998110744189, n_jobs=24, random_state=1)
print(cross_val_score(xgb, X, y, cv=kf, n_jobs=24, scoring='neg_mean_squared_error').mean())
xgb.fit(X,y)
explainer = shap.TreeExplainer(xgb)
shap_value = pd.DataFrame(explainer.shap_values(X), columns=X.columns)
shap_interaction_values = shap.TreeExplainer(xgb).shap_interaction_values(X)

# Lasso
model = LassoCV(cv=10).fit(X, y)
importance = np.abs(model.coef_)
print([X.columns[idx] for idx, val in enumerate(importance) if val > 0])

# Support vector machine
svr = SVR()
sfs = SequentialFeatureSelector(svr)
sfs.fit(X, y)
print("Features selected by forward sequential selection: "
      f"{X.columns[sfs.get_support()].tolist()}")

# Random forest
rf = RandomForestRegressor()
sfs = SequentialFeatureSelector(rf)
sfs.fit(X, y)
print(r2_score(y, rf.predict(X)))
print("Features selected by forward sequential selection: "
      f"{X.columns[sfs.get_support()].tolist()}")

rf = RandomForestRegressor(n_estimators=415, max_depth=8, max_features=7, n_jobs=12)
rf.fit(X,y)
explainer = shap.TreeExplainer(rf)
shap.summary_plot(explainer.shap_values(X), X, max_display=10)
imp_sort = pd.DataFrame(abs(explainer.shap_values(X)), columns=X.columns).mean().sort_values(ascending=False)
(imp_sort / imp_sort.sum())[(imp_sort / imp_sort.sum() * 100)>10]


exclude = pd.DataFrame()
for i in np.arange(f.CountryName.nunique()):
    f1 = f.loc[~f.CountryName.isin([f.CountryName.unique()[i]])]
    X = f1[['Lag_5_C1_sub_index', 'Lag_4_C2_sub_index', 'Lag_5_C3_sub_index',
           'Lag_3_C4_sub_index', 'C5_sub_index', 'Lag_1_C6_sub_index',
           'Lag_2_C7_sub_index', 'Lag_2_C8_sub_index', 'E1_sub_index',
           'E2_sub_index', 'Lag_7_H1_sub_index', 'H2_sub_index', 'Lag_4_H3_sub_index', 'Lag_4_H6_sub_index']]
    y = f1[['dif']]
    xgb = XGBRegressor(reg_alpha=0.0001004896123179, colsample_bytree=1, gamma=0.29999999999999993, reg_lambda=0.99144791324819, 
                       learning_rate=0.3027546451247583, max_depth=6, min_child_weight=1, n_estimators=114, subsample=0.9998110744189, n_jobs=24, random_state=1)
    xgb.fit(X,y)
    explainer = shap.TreeExplainer(xgb)
    exclude = pd.concat([exclude.reset_index(drop=True), 
                         pd.DataFrame(pd.DataFrame(abs(explainer.shap_values(X)), columns=X.columns).mean().sort_values(ascending=False).index[:5].values, columns=[f.CountryName.unique()[i]]).reset_index(drop=True)], axis=1)
imp_sort = pd.DataFrame(abs(explainer.shap_values(X)), columns=X.columns).mean().sort_values(ascending=False)
(imp_sort / imp_sort.sum())[(imp_sort / imp_sort.sum() * 100)>10].index
exclude.columns = f.CountryName.unique()
print(pd.Series(exclude.unstack().values).value_counts())


f1 = f.loc[~f.CountryName.isin(['Canada', 'China', 'United States of America', 'Russian Federation', ])]
X = f1[['Lag_5_C1_sub_index', 'Lag_4_C2_sub_index', 'Lag_5_C3_sub_index',
       'Lag_3_C4_sub_index', 'C5_sub_index', 'Lag_1_C6_sub_index',
       'Lag_2_C7_sub_index', 'Lag_2_C8_sub_index', 'E1_sub_index',
       'E2_sub_index', 'Lag_7_H1_sub_index', 'H2_sub_index', 'Lag_4_H3_sub_index', 'Lag_4_H6_sub_index']]
y = f1[['dif']]#*100
xgb = XGBRegressor(reg_alpha=0.0001004896123179, colsample_bytree=1, gamma=0.29999999999999993, reg_lambda=0.99144791324819, 
                   learning_rate=0.3027546451247583, max_depth=6, min_child_weight=1, n_estimators=114, subsample=0.9998110744189, n_jobs=24, random_state=1)
xgb.fit(X,y)
explainer = shap.TreeExplainer(xgb)
imp_sort = pd.DataFrame(abs(explainer.shap_values(X)), columns=X.columns).mean().sort_values(ascending=False)
(imp_sort / imp_sort.sum())[(imp_sort / imp_sort.sum() * 100)>10]


X = f[['Lag_5_C1_School closing', 'Lag_3_C2_Workplace closing', 'Lag_4_C3_Cancel public events',
       'Lag_3_C4_Restrictions on gatherings', 'C5_Close public transport',
       'Lag_1_C6_Stay at home requirements', 'C7_Restrictions on internal movement',
       'Lag_2_C8_International travel controls', 'E1_Income support',
       'E2_Debt/contract relief', 'Lag_7_H1_Public information campaigns',
       'H2_Testing policy', 'Lag_4_H3_Contact tracing', 'H6_Facial Coverings']]
y = f[['dif']]

xgb = XGBRegressor(reg_alpha=0, colsample_bytree=0.84, gamma=0.1175, reg_lambda=0.8, 
                   learning_rate=0.1, max_depth=10, min_child_weight=1, n_estimators=100, subsample=1, n_jobs=24, random_state=1)
print(cross_val_score(xgb, X, y, cv=kf, n_jobs=24, scoring='neg_mean_squared_error').mean())
xgb.fit(X,y)
explainer = shap.TreeExplainer(xgb)
imp_sort = pd.DataFrame(abs(explainer.shap_values(X)), columns=X.columns).mean().sort_values(ascending=False)
print((imp_sort / imp_sort.sum())[(imp_sort / imp_sort.sum() * 100)>10])
imp_sort = pd.DataFrame(abs(explainer.shap_values(X)), columns=X.columns).mean().sort_values(ascending=False)
imp_sort / imp_sort.sum() * 100
print(pd.Series(imp_sort / imp_sort.sum() * 100).cumsum())
