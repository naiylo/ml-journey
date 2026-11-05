# ------------ 2.5 Pipelines ------------

# Instead of manually preprocessing, we want to combine this into one single pipeline.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)                                                              # This time we will use a real-world dataset (569 samples, 30 features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

model = Pipeline([                                                                                      # First we create our model here you can include basically everything (more details below)
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)                                                                             # Then we train/fit our model on our training data

print("Train accuracy:", model.score(X_train, y_train))                                                 # Lastly we can validate the models accuracy
print("Test accuracy:", model.score(X_test, y_test))

# Obviusly you can put a lot of different steps into a model but here is a general oversight.
#
# model = Pipeline([
#     ("imputer", SimpleImputer(strategy="mean")),                      -> First we replace missing values with the mean
#     ("scaler", StandardScaler()),                                     -> Next we normalize each feature (mean=0, std=1)
#     ("features", PolynomialFeatures(degree=2)),                       -> (This is something we did not cover) Generate squared features and pairwise interactions
#     ("selection", SelectKBest(score_func=f_classif, k=5)),            -> (This is something we did not cover) Select the 5 features with the highest ANOVA F-scores
#     ("classifier", LogisticRegression(max_iter=1000))                 -> Train a logistic regression classifier
# ])

# You can also use a model in a model, for example to splitup the prepocessing of categorical and numerical features into two seperate pipelines.

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

numeric_pipeline = Pipeline([                                           # We first define a pipeline for all numerical values
    ("imputer", SimpleImputer(strategy="mean")),                        # Fill missing values with the mean
    ("scaler", StandardScaler())                                        # Then normalize each feature (mean=0, std=1)
])

categorical_pipeline = Pipeline([                                       # Then we define a pipeline for all categorical values
    ("imputer", SimpleImputer(strategy="most_frequent")),               # We replace missing values with the most frequent occuring category
    ("encoder", OneHotEncoder(handle_unknown="ignore"))                 # Then we encode the categories 
])

preprocessor = ColumnTransformer([                                      # We combine these two different prepocessing pipelines into one
    ("numeric", numeric_pipeline, ["age", "income"]),                   # Differentiating by feature type (For that we have to know the 
    ("categorical", categorical_pipeline, ["city", "gender"])           # categories of our data and define which is which)
])

model = Pipeline([                                                      # Lastly we can put this preprocessor in our general model
    ("preprocessor", preprocessor),                                     # First we preprocess based on feature type
    ("classifier", LogisticRegression())                                # Then we apply LR
])