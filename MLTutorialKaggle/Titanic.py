import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

X_train_full = pd.read_csv("train.csv", delimiter=",")
test_data = pd.read_csv("test.csv")

X_train_full.dropna(axis=0, subset=['Survived'], inplace=True)
y_train = X_train_full['Survived']
X_train_full.drop(['Survived'], axis=1, inplace=True)

X_test_full = test_data

#dealing with the non-numerical data
#hot encoding for sex, ticket, cabin and embarked cols
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == 'object']

numerical_cols = [cname for cname in X_train_full.columns if
                X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


#preprocess data
numerical_transformer = SimpleImputer(strategy='most_frequent')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

model = XGBRegressor(n_estimators=1000, learning_rate=0.01)

#Bundle preprocessing and modeling in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                      ])


clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
predictions = np.rint(predictions)
int_predictions = predictions.astype(int)

output = {'PassengerId': test_data.PassengerId,
                       'Survived': int_predictions}

df = pd.DataFrame(output)
print(df)

df.to_csv('submission.csv',index=False)
print("Your submission was successfully saved!")


























