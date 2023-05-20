import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle

# load data
df = pd.read_csv('diabetes_data.csv')

# split data into train and test sets
X = df.drop('HighBP', axis=1)
y = df['HighBP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define objective function
def objective(params):
    model = xgb.XGBClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        colsample_bytree=params['colsample_bytree'],
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    return {'loss': -roc_auc, 'status': STATUS_OK}

# define search space
search_space = {'n_estimators': hp.quniform('n_estimators', 50, 500, 1),
                'max_depth': hp.quniform('max_depth', 1, 10, 1),
                'learning_rate': hp.loguniform('learning_rate', -5, 0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)}

# perform hyperparameter tuning using Tree Parzen Estimator (TPE)
trials = Trials()
best_params = fmin(objective,
                   space=search_space,
                   algo=tpe.suggest,
                   max_evals=100,
                   trials=trials)

# train XGBoost model with best hyperparameters
model = xgb.XGBClassifier(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']),
    learning_rate=best_params['learning_rate'],
    colsample_bytree=best_params['colsample_bytree'],
    objective='binary:logistic',
    random_state=42
)
model.fit(X_train, y_train)

# evaluate performance of model on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Best Hyperparameters: ", best_params)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

# get feature importances of model
importances = model.feature_importances_
sorted_idx = importances.argsort()[::-1]

# print feature importances
for idx in sorted_idx:
    print("{:0.3f} {}".format(importances[idx], X.columns[idx]))

# save model
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))