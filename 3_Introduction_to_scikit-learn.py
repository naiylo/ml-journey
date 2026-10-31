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

# ------------ 2.2 Different splitting techniques ------------