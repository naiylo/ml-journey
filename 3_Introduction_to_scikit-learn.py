# ------------ 2.1 Creating train, validation and test sets ------------

# For splits the most essential method is train_test_split.

import numpy as np
from sklearn.model_selection import train_test_split

X = np.array([                                              # 20 samples, each containing 2 features: [hours_studied, hours_slept] <- your original dataset
    [1, 5], [2, 6], [3, 7], [4, 8], [5, 6],
    [6, 7], [7, 8], [8, 5], [9, 6], [10, 7],
    [2, 8], [3, 5], [4, 6], [5, 7], [6, 8],
    [7, 5], [8, 6], [9, 7], [10, 8], [1, 6]
])

X_train, X_temp = train_test_split(                         # Step 1: Split into training (70%) and temporary (30%)
    X,
    test_size=0.30,
    random_state=42
)

X_val, X_test = train_test_split(                           # Step 2: Split temporary data equally into validation and test
    X_temp,
    test_size=0.50,
    random_state=42
)

# Setting the random_state fixed makes this split reproducable, which is very important if you want to enable others to reproduce your results.

print("Training:", X_train.shape)    
print("Validation:", X_val.shape)     
print("Testing:", X_test.shape)       

print("\nTraining samples:\n", X_train)
print("\nValidation samples:\n", X_val)
print("\nTest samples:\n", X_test)

# There are also other methods for splitting your data, that are particulary useful in different scenarios. For example: StratifiedKFold for Classification with class balance,
# GroupShuffleSplit for Group-aware random train/test split, or GroupKFold if multiple samples are from the same group, or TimeSeriesSplit for Chronologically ordered data.

# ------------ 2.2 Scaling and normalizing features ------------

# For this topic in particular these imports are very useful.

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

# The most important difference is how these two options normalize the vector.

X = np.array([
    [10, 100],
    [20, 200],
    [30, 300]
])

# While the StandardScaler works column by column, effectively normalizing [[10,...],[20,...],[30,...]] and [[...,100],[...,200],[...,300]]
# The normalize function works row, by row, effectively normalizing [[10,100]] and [20,200] and [30,300] individually

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

X_normalized = normalize(X, norm="l2")        # We use L2 here but effetively you can also use other normalizations

print("Original:\n", X)
print("\nStandardScaler:\n", X_standardized.round(3))
print("\nL2 Normalization:\n", X_normalized.round(3))

# Both have there use in difference scenarios.
# StandardScaler: If you have data such as age, income, height ; Also for Logistic Regression, KNN, (Often) PCA
# Normalize: (Often) for embedding comparisons, For sentence embeddings

# ------------ 2.3 Handling missing values and categorical features ------------