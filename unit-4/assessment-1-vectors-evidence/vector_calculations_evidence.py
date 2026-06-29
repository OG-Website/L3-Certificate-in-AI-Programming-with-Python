import numpy as np

# Unit 4 Assessment 1 - Vector calculation evidence
# This file demonstrates vectors, transpose, magnitude, direction, addition,
# scalar multiplication and dot product using NumPy.

vector_a = np.array([4, 3])
vector_b = np.array([1, 2])

print("Vector A:", vector_a)
print("Vector B:", vector_b)
print("Vector A shape:", vector_a.shape)

row_vector = np.array([[4, 3]])
column_vector = row_vector.T
print("Row vector:")
print(row_vector)
print("Column vector after transpose:")
print(column_vector)
print("Row shape:", row_vector.shape)
print("Column shape:", column_vector.shape)

magnitude = np.linalg.norm(vector_a)
angle_radians = np.arctan2(vector_a[1], vector_a[0])
angle_degrees = np.degrees(angle_radians)
print("Magnitude of Vector A:", magnitude)
print("Direction of Vector A in degrees:", round(angle_degrees, 2))

vector_sum = vector_a + vector_b
scaled_vector = 2 * vector_a
dot_product = np.dot(vector_a, vector_b)
print("Vector A + Vector B:", vector_sum)
print("2 * Vector A:", scaled_vector)
print("Dot product of Vector A and Vector B:", dot_product)
