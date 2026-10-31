# ------------ 1.4 Useful functions (sum, mean, matrix operations) ------------

import numpy as np

# For example you can easily combine two arrays with the same shape by adding them together.

stacked_array_1 = np.array([1,2,3])
stacked_array_2 = np.array([4,5,6])

combined_stack = np.stack((stacked_array_1, stacked_array_2))                   # This would give us a 2x3 array
combined_stack_columns =np.column_stack((stacked_array_1, stacked_array_2))     # This would give us a 3x2 array

# We can also aggregate and sum the values inside of arrays. This is especially useful for machine learning, 
# when we want to calculate the loss of our model, or statistics such as false positives, false negatives, true positives, and true negatives.

sum_array = np.sum(stacked_array_1)                             # This would give us 6, because 1+2+3 = 6
sum_array = np.sum(combined_stack[:,1])                         # This would give us 7, because 2+5 = 7
mean = np.mean(combined_stack)                                  # This would give us 3.5, because (1+2+3+4+5+6)/6 = 3.5
standard_dv = np.std(combined_stack)                            # This would give us the standard deviation of the values in combined_stack
variance = np.var(combined_stack)                               # This would give us the variance of the values in combined_stack

dummy_array = np.array([[1,2,3],[4,5,6],[7,8,9]])
comparison_array = np.array([[1,0,3],[4,0,6],[7,0,9]])
true_predictions = np.sum(dummy_array == comparison_array)      # This would give us 6, because we have 6 correct predictions

# There is of course also fuctions such as: np.min(), np.max(), np.argmin(), np.argmax(), np.median(), np.percentile(), np.corrcoef(), np.cov() and many more.

# ------------ 1.5 Vectorized operations and matrix operations ------------

# With vectors (tensors) we can easily do operations on all the elements at once, without the need for loops. This is called vectorization and is one of the main reasons why numpy is so fast.

dummy_array = np.array([[1,2,3],[4,5,6]])
add_dummy_array = dummy_array + 1                       # This would give us [[2,3,4],[5,6,7]]
mul_dummy_array = dummy_array * 2                       # This would give us [[2,4,6],[8,10,12]]

# You can use that for standardization, which is a common preprocessing step in machine learning.

mean = dummy_array.mean()                             
std = dummy_array.std()

dummy_scalled = (dummy_array - mean) / std              # This standardizes our dummy_array (in normal circumstances our X_train)

# Matrix operations are also quite easy

matrix_a = np.array([[1,2],[3,4]])
matrix_b = np.array([[1,2],[3,4]])

mu_matrix = matrix_a @ matrix_b                         # This would give us [[7,10],[15,22]] normal matrix multiplication
mu_matrix_2 = matrix_a * matrix_b                       # This would give us [[1,4],[9,16]] element wise multiplication

# These are the fundamental concepts and capabilities you should be familiar with for now. As you progress through the project examples, you will encounter more advanced functions 
# and different implementations. The smaller code snippets introduced here will provide a solid foundation, helping you apply and expand your knowledge in more complex scenarios.
