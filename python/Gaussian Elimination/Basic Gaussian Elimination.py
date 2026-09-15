import numpy as np

def gaussian_elimination(A: np.ndarray, b: np.ndarray, rounding_precision: int = 50) -> np.ndarray:
    """
    Solves the system of linear equations Ax = b using Gaussian elimination.

    Parameters:
    A (numpy.ndarray): Coefficient matrix of shape (n, n).
    b (numpy.ndarray): Right-hand side vector of shape (n,).
    rounding_precision (int): Number of digits after the decimal point of division results. Needed for comparing truncation errors when pivoting is used.

    Returns:
    numpy.ndarray: Solution vector x of shape (n,).
    """
    # Basic input validation
    if A.shape[0] != A.shape[1]:
        print("Matrix A must be square.")
        return
    if A.shape[0] != b.shape[0] or b.shape[1] != 1:
        # print(A.shape[0])
        # print(b.shape[0])
        b.shape[1]
        print("Vector b must have the same number of rows as A and be one-dimensional.")
        return

    # Create and Augmented, Upper Triangular Matrix
    n = A.shape[0]

    augmented_matrix = np.concatenate((A, b), axis=1, dtype=float)

    for j in range(0, n):
        for i in range(j+1, n):
            if augmented_matrix[j][j] == 0: 
                print("zero divsion error")
                return
            scaling_factor = round(augmented_matrix[i][j] / augmented_matrix[j][j], rounding_precision)

            augmented_matrix[i] = augmented_matrix[i] - scaling_factor * augmented_matrix[j]
        # print(augmented_matrix)

    #Backwards substitution
    x = np.zeros(n)
    x[n-1] = round(augmented_matrix[n-1][n] / augmented_matrix[n-1][n-1], rounding_precision)

    for i in range(n-2, -1, -1):
        x[i] = augmented_matrix[i][n]
        for j in range(n-1, i, -1):
            x[i] -= x[j] * augmented_matrix[i][j]

        x[i] = round(x[i] / augmented_matrix[i][i], rounding_precision)
    # print(x)

    #Check the results
    y = np.zeros(n)
    e = np.zeros(n)

    for i in range(0, n):
        for j in range(0,n):
            y[i] += A[i][j] * x[j]
        e[i] = y[i] - b[i][0]

    print(f"y = {y},\n b = {b},\n abs error = {e}")

            
#Testing
test_cases = [
    # 1. Basic 3x3 system
    (
        np.array([
            [1,  1, 3],
            [0,  1, 3],
            [-1, 3, 0]
        ], dtype=float),
        np.array([1, 3, 5], dtype=float)
    ),

    # 2. Simple 2x2 system
    (
        np.array([
            [2, 1],
            [1, 3]
        ], dtype=float),
        np.array([5, 6], dtype=float)
    ),

    # 3. 1x1 system
    (
        np.array([
            [4]
        ], dtype=float),
        np.array([8], dtype=float)
    ),

    # 4. Identity matrix
    (
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float),
        np.array([4, -2, 7], dtype=float)
    ),

    # 5. Diagonal matrix
    (
        np.array([
            [2, 0, 0],
            [0, -4, 0],
            [0, 0, 5]
        ], dtype=float),
        np.array([6, 8, 15], dtype=float)
    ),

    # 6. Upper triangular matrix
    (
        np.array([
            [2, 3, -1],
            [0, 4,  2],
            [0, 0,  5]
        ], dtype=float),
        np.array([5, 6, 10], dtype=float)
    ),

    # 7. Lower triangular matrix
    (
        np.array([
            [2, 0, 0],
            [3, 4, 0],
            [1, 2, 5]
        ], dtype=float),
        np.array([4, 11, 13], dtype=float)
    ),

    # 8. Requires a row swap immediately because A[0, 0] = 0
    (
        np.array([
            [0, 2],
            [1, 3]
        ], dtype=float),
        np.array([4, 5], dtype=float)
    ),

    # 9. Requires pivoting later in the elimination
    (
        np.array([
            [1, 1, 1],
            [2, 2, 5],
            [4, 6, 8]
        ], dtype=float),
        np.array([3, 9, 18], dtype=float)
    ),

    # 10. Negative coefficients
    (
        np.array([
            [-2,  3, -1],
            [ 4, -1,  2],
            [-3,  2,  5]
        ], dtype=float),
        np.array([1, 7, -4], dtype=float)
    ),

    # 11. Floating-point coefficients
    (
        np.array([
            [0.5,  1.2, -0.3],
            [2.1, -0.7,  1.5],
            [1.0,  0.4,  2.2]
        ], dtype=float),
        np.array([2.4, -1.3, 5.1], dtype=float)
    ),

    # 12. Very different coefficient magnitudes
    # Useful for seeing numerical/pivoting problems.
    (
        np.array([
            [1e-10, 1],
            [1,     1]
        ], dtype=float),
        np.array([1, 2], dtype=float)
    ),

    # 13. 4x4 system
    (
        np.array([
            [ 2,  1, -1,  2],
            [ 4,  5, -3,  6],
            [-2,  5, -2,  6],
            [ 4, 11, -4,  8]
        ], dtype=float),
        np.array([5, 9, 4, 2], dtype=float)
    ),

    # 14. 5x5 system
    (
        np.array([
            [ 4,  1,  0, -1,  2],
            [ 1,  5,  2,  0, -1],
            [ 0,  2,  6,  1,  1],
            [-1,  0,  1,  7,  2],
            [ 2, -1,  1,  2,  8]
        ], dtype=float),
        np.array([6, 7, 8, 9, 10], dtype=float)
    ),

    # ---------------- EDGE CASES ----------------

    # 15. Singular matrix: two rows are multiples
    # No unique solution.
    (
        np.array([
            [1, 2],
            [2, 4]
        ], dtype=float),
        np.array([3, 6], dtype=float)
    ),

    # 16. Singular matrix with inconsistent RHS
    # No solution.
    (
        np.array([
            [1, 2],
            [2, 4]
        ], dtype=float),
        np.array([3, 7], dtype=float)
    ),

    # 17. Singular 3x3 matrix
    # Third row = first row + second row.
    (
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [5, 7, 9]
        ], dtype=float),
        np.array([6, 15, 21], dtype=float)
    ),

    # 18. Entire zero row
    (
        np.array([
            [1, 2, 3],
            [0, 0, 0],
            [4, 5, 6]
        ], dtype=float),
        np.array([6, 0, 15], dtype=float)
    ),

    # 19. Zero matrix
    (
        np.array([
            [0, 0],
            [0, 0]
        ], dtype=float),
        np.array([0, 0], dtype=float)
    ),

    # 20. Zero matrix with nonzero RHS
    # Clearly inconsistent.
    (
        np.array([
            [0, 0],
            [0, 0]
        ], dtype=float),
        np.array([1, 0], dtype=float)
    ),

    # 21. Nearly singular matrix
    # Can reveal numerical-stability problems.
    (
        np.array([
            [1, 1],
            [1, 1 + 1e-12]
        ], dtype=float),
        np.array([2, 2 + 1e-12], dtype=float)
    ),

    # 22. Tiny first pivot
    # Partial pivoting should swap the rows.
    (
        np.array([
            [1e-15, 1, 1],
            [1,     2, 3],
            [2,     1, 1]
        ], dtype=float),
        np.array([2, 6, 4], dtype=float)
    ),

    # 23. Negative 1x1 case
    (
        np.array([
            [-5]
        ], dtype=float),
        np.array([20], dtype=float)
    ),

    # 24. RHS is all zeros
    # For an invertible A, solution should be the zero vector.
    (
        np.array([
            [3, 1, 2],
            [1, 4, 1],
            [2, 1, 5]
        ], dtype=float),
        np.array([0, 0, 0], dtype=float)
    ),
]

i=1
for (A, b) in test_cases:
    print(f"test {i}")
    print(A)
    b = np.atleast_2d(b).T
    print(b)
    gaussian_elimination(A, b)
    i += 1

# TODO: Compare with pivoting and without pivoting with rounding precision < 50 to see if truncation errors are significant.
# TODO: Deal with edge cases like singular matrices, inconsistent systems, and nearly singular matrices.
# TODO: Add functionality to handle cases where the system has no unique solution (e.g., infinite solutions or no solution).