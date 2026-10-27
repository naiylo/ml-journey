# ------------ 1.1 Creating arrays and inspecting data ------------
# Numpy is the fundamental package for working with datasets, and implementing mathematical operations.

# We always do that because it is more handy to write 'np' instead of 'numpy'.

import numpy as np

# This is how we create arrays in numpy.

a = np.array([1,2,3])               # This is 1-dimensional array (or vector) 
X = np.array([[1,2],[3,4],[5,6]])   # This is 3-dimensional array (or matrix)

# These are called "tensors" in deep learning. They are just multi-dimensional arrays. 
# The number of dimensions is called the "rank" of the tensor. The shape of a tensor is 
# a tuple of integers that describes the size of each dimension.

# You can also use numpy to intialize arrays with zeros, ones, identity matrices, or random numbers.

np.zeros((3,4))                 # 3x4 array of zeros
np.ones((3,4))                  # 3x4 array of ones
np.eye(3)                       # 3x3 identity matrix
np.random.random((3,4))         # 3x4 array of random numbers between 0 and 1

# However, numpy is not only useful for arrays, but also for sequences of numbers.

np.arange(0,10,2)               # This is a sequence of numbers from 0 to 10 with step 2
np.linspace(0,1,5)              # This is a sequence of 5 numbers from 0 to 1

# We can also use numpy to inspect our created arrays.

X.shape                 # This would give use (3,2), because we have 3 rows and 2 columns
X.ndim                  # This would give us 2, because we have 2 dimensions
X.size                  # This would give us the total elements in X, which would be 6
X.dtype                 # This would give us the data type of the elements in X, so int64

# ------------ 1.2 Indexing and slicing arrays ------------
# Indexing and slicing arrays is very similar to lists in Python. We also utilize numpy.

X[0]                # This would give us the first row of X
X[-1]               # This would give us the last row of X
X[0,1]              # This would give us the first row and second column of X, which is 2
X[[0,2]]            # This would give us the first and third rows of X, so [[1,2],[5,6]]
X[:2]               # This would give us the first two rows of X
X[0:2]              # You can also do that, but remeber that the last index is not included
X[:,0:1]            # This would give us the first feature of all the rows, so [[1],[3],[5]]

# Beyond these simple indexing and slicing, we can also utilize conditions and masks.

X[X[:,0] == 1]                          # This would give us all the rows where the first feature is equal to 1
mask = (X[:,0] > 1) & (X[:,1] < 6)      # First feature > 1, but second feature < 6
filtered_X = X[mask]                    # This would give us only [[3,4]]

# Especialy interesting for machine learning usage is this pattern.

correct_matrix = np.array([[1,1],[3,3],[5,5]])  # This way we can find wrong predictions
wrong_indices = np.where(correct_matrix != X)   # This gives us the wrong values in X at that indices

# Dont be confused by the output of correct_matrix ((array([0, 1, 2]), array([1, 1, 1]))).
# It is once again a matrix that means the errors where at [0,1], [1,1], and [2,1] in X.

# ------------ 1.3 Reshaping and combining arrays ------------
# This is where the error analysis begins, but we still only use numpy.

X.reshape(-1,6)         # This reshapes the matrix X to a 1x6 matrix, so it would give us [[1,2,3,4,5,6]]

# The -1 means that we let numpy automaticall calculate the number.
# Be aware that the total number of elements must remain the same, so no 2x4 out of a 3x2 matrix.
# But later on we will learn how to expand the number of dimensions.

X.T                      # This transposes the matrix X, so it would give us once again [[1,3,5],[2,4,6]]