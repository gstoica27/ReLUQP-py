from copy import deepcopy
import numpy as np



def ihlqr(A, B, Q, R, Qf, max_iters=1000, tol=1e-8):
    P = Qf
    K = np.zeros((B.shape[1], A.shape[0]))
    for _ in range(max_iters):
        P_prev = deepcopy(P)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
        if np.linalg.norm(P - P_prev, 2) < tol:
            return K, P
    raise Exception("ihlqr didn't converge")






def constrained_ihlqr(A, B_u, B_λ, C, Q, R, F, Qf, max_iters=1000, tol=1e-8):
    nu, nλ = B_u.shape[1], B_λ.shape[1]
    P = Qf
    K = np.zeros((B_u.shape[1], A.shape[0]))
    L = np.zeros((B_λ.shape[1], A.shape[0]))
    for k in range(max_iters):
        P_prev = deepcopy(P)
        kkt_lhs = np.block([[R + B_u.T @ P @ B_u, B_u.T @ P @ B_λ, B_u.T @ C.T],
                            [B_λ.T @ P @ B_u, F + B_λ.T @ P @ B_λ, B_λ.T @ C.T],
                            [C @ B_u, C @ B_λ, np.zeros((12, 12))]])
        kkt_rhs = np.block([B_u.T @ P @ A, B_λ.T @ P @ A, C @ A])
        assert np.linalg.matrix_rank(kkt_lhs) == kkt_lhs.shape[0], f"{np.linalg.matrix_rank(kkt_lhs)} {kkt_lhs.shape[0]}"
        if np.linalg.cond(kkt_lhs) > 1e11:
            print(f"KKT is ill-conditioned: {np.linalg.cond(kkt_lhs)}")
        gains = np.linalg.solve(kkt_lhs, kkt_rhs)
        K = gains[:nu, :]
        L = gains[nu:(nu+nλ), :]
        N = gains[(nu+nλ):, :]
        Ā = A - B_u @ K - B_λ @ L
        P = Q + A.T @ P @ Ā - A.T @ C.T @ N
        if np.linalg.norm(P - P_prev, 2) < tol:
            return K, L, P
        elif k == max_iters - 1:
            print(np.linalg.norm(P - P_prev, 2))
    raise Exception("ihlqr didn't converge")







def gen_sparse_mpc_qp(Ad, Bd, Q, R, Qf, horizon, A_add=None, l_add=None, u_add=None):
    nx, nu = Ad.shape[0], Bd.shape[1]
    H = np.block([[np.block([[R, Q]]) for _ in range(horizon - 1)], [R, Qf]])
    g = np.zeros(H.shape[0])
    A = np.kron(np.eye(horizon), np.block([[Bd, -np.eye(nu)]]))
    A[nx:, nu:(nu+nx)] += np.kron(np.eye(horizon - 1), np.block([[Ad, np.zeros((nx, nu))]]))
    l = np.zeros(A.shape[0])
    u = np.zeros(A.shape[0])
    if A_add is not None:
        A = np.vstack([A, A_add])
        l = np.hstack([l, l_add])
        u = np.hstack([u, u_add])
    return H, g, A, l, u









def gen_condensed_mpc_qp(Ad, Bd, Q, R, Qf, horizon, A_add, l_add, u_add, K=None):
    nx, nu = Ad.shape[0], Bd.shape[1]
    if K is None:
        K = np.zeros((nu, nx))
    H_sp, g_sp, _ = gen_sparse_mpc_qp(Ad, Bd, Q, R, Qf, horizon)
    F = np.kron(np.eye(horizon), np.block([[np.eye(nx)], [Bd]]))
    for k in range(1, horizon):
        F += np.kron(np.diag(np.ones(horizon - k), -k), np.block([[-K @ np.linalg.matrix_power(Ad - Bd @ K, k - 1) @ Bd], [(Ad - Bd @ K) @ np.linalg.matrix_power(Ad - Bd @ K, k) @ Bd]]))
    G = np.vstack([[-K @ np.linalg.matrix_power(Ad - Bd @ K, k - 1)], [(Ad - Bd @ K) @ np.linalg.matrix_power(Ad - Bd @ K, k)]] for k in range(1, horizon + 1))
    H = F.T @ H_sp @ F
    g_x0 = F.T @ H_sp @ G
    g = g_x0 @ np.zeros(nx) + F.T @ g_sp
    A = A_add @ F
    lu_x0 = -A_add @ G
    return H, g, A, l_add, u_add, g_x0, lu_x0













