# Reviewing this as well so I can then build the Tensor class for micrograd
# https://www.youtube.com/watch?v=VXU4LSAQDSc

import numpy as np
version = np.__version__

my_list = [1, 2, 3, 4]
print(my_list)

array = np.array([1, 2, 3, 4])
print(array)
print(type(array))

array *= 2
print(array)

# Multidimensional Arrays

array_0d = np.array('A') # O-dimensional array
print(array_0d.ndim)

array_1d = np.array(['A', 'B', 'C'])
print(array_1d.ndim)

array_2d = np.array([['A', 'B', 'C'],
                    ['D', 'E', 'F'],
                    ['G', 'H', 'I']])
print(array_2d.ndim)

array_3d = np.array([[['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']],
                     [['J', 'K', 'L'], ['M', 'N', 'O'], ['P', 'Q', 'R']],
                    [['S', 'T', 'U'], ['V', 'W', 'X'], ['Y', 'Z', ' ']]])
print(array_3d.ndim)
print(array_3d.shape) #dimensions are ranked in order by outer brackets first

print(array_3d[0][0][0]) #chain indexing
print(array_3d[0,0,0]) #multidimensional indexing

word = array_3d[1,1,1] + array_3d[0,2,2] + array_3d[2,0,0] + array_3d[0,2,1]
print(word) # NISH

# SLICING in NumPy

array_slice = np.array([[1, 2, 3, 4], 
                        [5, 6, 7, 8], 
                        [9, 10, 11, 12], 
                        [13, 14, 15, 16]])

# array[start:end:step]

'''
print(array_slice[0])
print(array_slice[-1]) # last row
print(array_slice[1:])


# Steps
# print(array_slice[0:4:2])
# print(array_slice[::2]) # returns first and third row

# print(array_slice[::-1]), returns all of the rows reversed.

# Column selection

print(array_slice[:, 0]) #returns column 1 values
print(array_slice[:, 1]) #returns column 2 values
print(array_slice[:, -1]) #returns last columns (column 4)

print(array_slice[:, 0:3]) #prints the first three columns
print(array_slice[:, 1:])

print(array_slice[:, 1::2]) #prints the 2nd and 4th column

print(array_slice[:, ::-1]) #prints columns in reverse order
'''

#print(array_slice[0:2, 0:2]) #selects the first two rows and first two columsn
print(array_slice[:2,2:])


# SCALAR ARITHMETIC

array_scalar = np.array([1, 2, 3])

print(array_scalar + 1)
print(array_scalar - 2)
print(array_scalar * 3)
print(array_scalar / 4) #converts to floats if necessary
print(array_scalar ** 5)


# VECTORIZED MATH FUNCTIONS

array_vector = np.array([1.01, 2.5, 3.99])

print(np.sqrt(array_vector))
print(np.round(array_vector)) # if 0.5, it rounds down. 
print(np.floor(array_vector))
print(np.ceil(array_vector))
print(np.pi)


# ELEMENT-WISE ARITHMETIC

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

print(array1 + array2) #vector product
print(array1 - array2)
print(array1 * array2)
print(array1 / array2)
print(array1 ** array2)


# COMPARISON OPERATORS

scores = np.array([91, 55, 100, 73, 82, 64])

print(scores == 100) # [False False  True False False False]
print(scores >= 60)  # [ True False  True  True  True  True]
print(scores < 60)   # [False  True False False False False]

scores[scores < 60] = 0
print(scores)


# BROADCASTING
# allows NumPy to perform operations on arrays with different shapes by virtually expanding dimensions
# so they match the larger array's shape

# The dimensions have the same size
# OR 
# One of the dimensions has a size of 1

arrayA = np.array([[1, 2, 3, 4]])
arrayB = np.array([[1], [2], [3], [4]])

print(arrayA.shape)
print(arrayB.shape)
#read dimensions right to left

print(arrayA * arrayB)


# AGGREGATE FUNCTIONS
# summarize data and typically return a single value

arrayX = np.array([[1, 2, 3, 4, 5], 
                  [6, 7, 8, 9, 10]])

print(np.sum(arrayX))
print(np.mean(arrayX))
print(np.std(arrayX))
print(np.var(arrayX))
print(np.min(arrayX))
print(np.max(arrayX))
print(np.argmin(arrayX))
print(np.argmax(arrayX))
print(np.sum(arrayX, axis=0))
print(np.sum(arrayX, axis=1))


# FILTERING
# selecting elements from an array that meet a certain condition

ages = np.array([[21, 17, 19, 20, 16, 30, 18, 65], 
                 [39, 22, 15, 99, 18, 19, 20, 21]])

teenagers = ages[ages < 18]
print(teenagers)

adults = ages[(ages >= 18) & (ages < 65)]
print(adults)

seniors = ages[ages >= 65]
print(seniors)

evens = ages[ages % 2 == 0]
print(evens)

odds = ages[ages % 2 == 1]
print(odds)

adults = np.where(ages >= 18, ages, np.nan)
# np.where(condition, array, fill value)
print(adults)


# RANDOM NUMBERS

rng = np.random.default_rng(seed = 1)
print(rng.integers(1, 7)) #second number is exclusive
print(rng.integers(low=1, high=101))
print(rng.integers(1, 11, 5)) #will store 5 numbers in a 1D array
print(rng.integers(1, 101, size=(3,2)))

np.random.seed(seed=1)

print(np.random.uniform()) #random floating point between 0 and 1
print(np.random.uniform(-1, 1))
print(np.random.uniform(-1, 1, (2,3)))

# shuffling an array

rng2 = np.random.default_rng(seed = 2)
arrayShuffle = np.array([1, 2, 3, 4, 5])
print(arrayShuffle)
rng2.shuffle(arrayShuffle)
print(arrayShuffle)

# random choice?
fruits = np.array(["apple", "orange", "banana", "coconut", "pineapple"])
print(fruits)

fruit = rng2.choice(fruits)
print(fruit)

fruits_random = rng.choice(fruits, size=(3,3))
print(fruits_random)