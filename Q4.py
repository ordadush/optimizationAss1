import numpy as np

def solve_weighted_least_squares_normal_eqs(A: np.ndarray, b: np.ndarray, lamb: np.ndarray):
    At = A.T
    AtA = At @ A
    Atb = At @ b
    x = np.linalg.solve(AtA + lamb, Atb)
    return x

def main():
    A = np.array([[2, 1, 2],[1, -2, 1],[1, 2, 3],[1, 1, 1]
    ], dtype=float)
    b = np.array([6, 1, 5, 2], dtype=float)
    lamb = np.diag([0.5, 0.5,0.5])  # lambda is a diagonal matrix 0.5
    
    x = solve_weighted_least_squares_normal_eqs(A, b, lamb)

    print("solution", x) #[1.38517367 0.53499222 0.88958009]


if __name__ == "__main__":
    main()
