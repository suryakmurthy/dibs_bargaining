import torch
import cvxpy as cp
import numpy as np
from scipy.optimize import approx_fprime
from helper_functions import from_subspace_to_simplex, from_simplex_to_subspace, project_gradient_to_simplex_tangent, compute_simplex_gradient, angle_between
from comparison_based_estimation import estimate_gradient_f_i_comparisons, gradient_estimation_sign_opt_batch
import concurrent.futures
import time


def solve_markowitz(Sigma: torch.Tensor, lambda_mu: torch.Tensor):
    """
    Solve Markowitz optimization using convex programming with simplex constraints.
    Returns: tensor of weights if successful, otherwise None.
    """

    Sigma_np = Sigma.detach().cpu().numpy().astype(np.float64)
    lambda_mu_np = lambda_mu.detach().cpu().numpy().astype(np.float64)
    n = len(lambda_mu_np)
    w = cp.Variable(n)

    # Diagnostics
    eigvals = np.linalg.eigvalsh(Sigma_np)
    min_eig = np.min(eigvals)
    cond_number = np.max(eigvals) / (min_eig + 1e-12)
    # print("Min eigenvalue:", min_eig)
    # print("Condition number:", cond_number)

    # Define optimization problem
    objective = cp.Minimize(cp.quad_form(w, Sigma_np) - lambda_mu_np @ w)
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except cp.error.SolverError as e:
        print("SolverError:", str(e))
        return None

    # Handle solver failure or infeasibility
    if w.value is None or problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Solver status: {problem.status}")
        return None

    return torch.tensor(w.value, dtype=Sigma.dtype)

def solve_markowitz_subspace_barrier(Sigma: torch.Tensor, lambda_mu: torch.Tensor, barrier_coeff=1e-6):
    """
    Solve Markowitz using subspace log-barrier formulation.
    """
    Sigma_np = Sigma.detach().cpu().numpy()
    lambda_mu_np = lambda_mu.detach().cpu().numpy()
    n = len(lambda_mu_np)
    v = cp.Variable(n - 1)
    last_coord = 1 - cp.sum(v)
    w = cp.hstack([v, last_coord])

    barrier = -cp.sum(cp.log(v)) - cp.log(1 - cp.sum(v))
    objective = cp.Minimize(cp.quad_form(w, Sigma_np) - lambda_mu_np @ w + barrier_coeff * barrier)
    constraints = [v >= 1e-8, cp.sum(v) <= 1 - 1e-8]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if v.value is None:
        raise ValueError("Optimization failed.")

    v_tensor = torch.tensor(v.value, dtype=Sigma.dtype)
    w_tensor = torch.cat([v_tensor, 1.0 - torch.sum(v_tensor).unsqueeze(0)], dim=0)
    return w_tensor

def solve_nbs_first_order_simplex(Sigma_list, lambda_mu_list, starting_point=None,
                                   disagreement=-1.0, steps=1000, lr=0.01):
    """
    First-order solver for Nash Bargaining in (n-1) space.
    """
    m = len(Sigma_list)
    n = Sigma_list[0].shape[0]
    w = torch.ones(n, dtype=torch.float64) * (1.0 / n) if starting_point is None else starting_point.clone()
    state_progression = [w.tolist()]
    for step in range(steps):
        grad_sum = torch.zeros_like(w)
        gradients = []
        nash_values = []
        for i in range(m):
            u_i = torch.dot(lambda_mu_list[i], w) - torch.dot(w, Sigma_list[i] @ w)
            if u_i <= disagreement:
                continue
            grad_i = -1 * compute_simplex_gradient(w, Sigma_list[i], lambda_mu_list[i])
            projected_gradient_i = project_gradient_to_simplex_tangent(grad_i)
            gradients.append(projected_gradient_i)
            nash_values.append(projected_gradient_i / (u_i - disagreement + 1e-8))
            # if step == 0:
            #     print("Double - Checking this: ", u_i, u_i - disagreement + 1e-8)
            grad_sum += projected_gradient_i / (u_i - disagreement + 1e-8)

        grad_norm = grad_sum.norm()
        # if step == 0:
        #     print(f"Checking Gradients for Nash Solution at step {step}: {nash_values}")
        w_next = w + (lr * (grad_sum / grad_norm))
        while ((w_next < 0).any()) and lr > 1e-12:
            lr *= 0.1
            w_next = w + (lr * (grad_sum / grad_norm))
            # print("Checking Values: ", w_next, lr, (grad_sum / grad_norm))
        if lr <= 1e-12:
            break
        # print("Solved!")
        state_progression.append(w_next.tolist())
        w = w_next.detach().clone().requires_grad_(True)

    return w.detach(), state_progression

def solve_nbs_cvxpy(Sigma_list, lambda_mu_list, disagreement=-1.0):
    """
    Solve Nash Bargaining analytically with log utilities using CVXPY.
    """
    m = len(Sigma_list)
    n = Sigma_list[0].shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]

    exprs = []
    for i in range(m):
        util = lambda_mu_list[i] @ w - cp.quad_form(w, Sigma_list[i])
        exprs.append(cp.log(cp.reshape(util - disagreement, ())))

    objective = cp.Maximize(cp.sum(exprs))
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver=cp.SCS)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Problem status:", problem.status)
        return None

    return w.value

def estimate_gradient_fd_simplex(w, Sigma, lambda_mu, epsilon=1e-5):
    w_np = w.detach().numpy()
    query_counter = {'count': 0}

    def wrapped_fn(w_input_np):
        query_counter['count'] += 1
        w = torch.tensor(w_input_np, dtype=torch.float64)

        quad = torch.dot(w, Sigma @ w)
        linear = torch.dot(lambda_mu, w)
        loss = quad - linear

        return loss.item()

    grad_np = approx_fprime(w_np, wrapped_fn, epsilon)
    grad_est = torch.tensor(grad_np, dtype=w.dtype)
    grad_true = compute_simplex_gradient(w, Sigma, lambda_mu)
    grad_error = torch.norm(grad_est - grad_true).item()
    return grad_est, query_counter['count'], grad_error

def solve_nbs_zeroth_order(Sigma_list, lambda_mu_list, starting_point=None,
                           disagreement=-1.0, steps=1000, lr=0.01):
    m = len(Sigma_list)
    n = Sigma_list[0].shape[0]
    w = torch.ones(n, dtype=torch.float64) * (1.0 / n) if starting_point is None else starting_point.clone()
    total_queries = 0
    grad_errors = []

    state_progression = [w.tolist()]

    for step in range(steps):
        grad_sum = torch.zeros_like(w)
        for i in range(m):
            u_i = torch.dot(lambda_mu_list[i], w) - torch.dot(w, Sigma_list[i] @ w)
            if u_i <= disagreement:
                continue
            grad_i, query_count, grad_error = estimate_gradient_fd_simplex(w, Sigma_list[i], lambda_mu_list[i])

            grad_i = -1 * project_gradient_to_simplex_tangent(grad_i)
            grad_sum += grad_i / (u_i - disagreement + 1e-8)
            total_queries += query_count
            grad_errors.append(grad_error)

        grad_norm = grad_sum.norm()
        if grad_norm > 0:
            w_next = w + (lr * (grad_sum / grad_norm))
            while ((w_next < 0).any()) and lr > 1e-12:
                lr *= 0.1
                w_next = w + (lr * (grad_sum / grad_norm))
            if lr <= 1e-12:
                break
            state_progression.append(w_next.tolist())
            w = w_next.detach().clone().requires_grad_(True)

    avg_grad_error = np.mean(grad_errors) if grad_errors else 0.0
    return w.detach(), total_queries, state_progression

def solve_nbs_barrier(Sigma_list, lambda_mu_list, disagreement=-1.0, barrier_coeff=1e-6):
    """
    Solve Nash Bargaining using subspace log-barrier directly in CVXPY.
    """
    m = len(Sigma_list)
    n = Sigma_list[0].shape[0]
    v = cp.Variable(n - 1)
    w = cp.hstack([v, 1 - cp.sum(v)])

    exprs = [cp.log(cp.reshape(lambda_mu_list[i] @ w - cp.quad_form(w, Sigma_list[i]) - disagreement, ()))
             for i in range(m)]

    barrier_terms = cp.sum(cp.log(v)) + cp.log(1 - cp.sum(v))
    total_obj = cp.sum(exprs) + barrier_coeff * barrier_terms

    objective = cp.Maximize(total_obj)
    problem = cp.Problem(objective)
    result = problem.solve()

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Problem status:", problem.status)
        return None

    return np.append(v.value, 1 - np.sum(v.value))

def run_our_solution_concept_actual(x0, Sigma_set, lambda_mu_set, x_i_set, steps=1000, step_size=0.01):
    """
    Run iterative update minimizing distance to all agents' optima.
    """
    x = x0.clone().detach().requires_grad_(True)
    state_progression = [x.tolist()]
    for s in range(steps):
        grad_sum = torch.zeros_like(x)
        norm_sum = 0.0
        projected_gradients = []
        norm_values = []
        for j in range(len(lambda_mu_set)):
            grad = compute_simplex_gradient(x, Sigma_set[j], lambda_mu_set[j])
            grad_projected = project_gradient_to_simplex_tangent(grad)
            projected_gradients.append(grad_projected)
            x_opt = x_i_set[j]
            grad_norm = torch.norm(grad_projected)
            x_diff_norm = torch.norm(x - x_opt)

            if grad_norm > 0:
                grad_sum += (grad_projected / grad_norm) * x_diff_norm
                norm_values.append((grad_projected / grad_norm) * x_diff_norm)
                norm_sum += x_diff_norm
        # if s == 0:
        #     print(f"Checking Gradients for Nash Solution at step {s}: {norm_values}")
        x_new = x - step_size * (grad_sum / norm_sum)
        while ((x_new < 0).any()) and step_size > 1e-12:
            step_size *= 0.1
            x_new =  x - step_size * (grad_sum / norm_sum)
        if step_size <= 1e-12:
            break
        state_progression.append(x_new.tolist())
        x = x_new.detach().clone().requires_grad_(True)

    return x, state_progression

def run_our_solution_concept_comparisons(x0, Sigma_set, lambda_mu_set, x_i_set, steps=1000, step_size=0.01):
    """
    Run iterative update using comparison-based gradient estimation.
    """
    x = x0.clone()
    total_queries = 0

    state_progression = [x.tolist()]
    for _ in range(steps):
        grad_sum = 0
        norm_sum = 0
        for j in range(len(lambda_mu_set)):
            grad, query_count = estimate_gradient_f_i_comparisons(x, Sigma_set[j], lambda_mu_set[j])
            total_queries += query_count
            x_opt = x_i_set[j]
            grad_sum += (grad / torch.norm(grad)) * torch.norm(x - x_opt)
            norm_sum += torch.norm(x - x_opt)

        x_new = x - step_size * (grad_sum / norm_sum)
        while ((x_new < 0).any()) and step_size > 1e-12:
            step_size *= 0.1
            x_new =  x - step_size * (grad_sum / norm_sum)
        if step_size <= 1e-12:
            break
        state_progression.append(x_new.tolist())
        x = x_new.detach().clone().requires_grad_(True)
    return x, total_queries

def run_our_solution_concept_comparisons_parallel(x0, Sigma_set, lambda_mu_set, x_i_set, steps=1000, step_size=0.01):
    """
    Run iterative update using comparison-based gradient estimation.
    Parallelizes gradient estimation across agents.
    """
    x = x0.clone().detach().requires_grad_(True)
    total_queries = 0

    for step in range(steps):
        grad_sum = torch.zeros_like(x)
        norm_sum = 0.0

        def estimate_for_agent(j):
            grad, query_count = estimate_gradient_f_i_comparisons(x, Sigma_set[j], lambda_mu_set[j])
            x_opt = x_i_set[j]
            diff_norm = torch.norm(x - x_opt)
            grad_unit = grad / torch.norm(grad)
            return grad_unit * diff_norm, diff_norm, query_count

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(estimate_for_agent, j) for j in range(len(lambda_mu_set))]
            for future in concurrent.futures.as_completed(futures):
                try:
                    grad_contrib, norm_contrib, query_count = future.result()
                    grad_sum += grad_contrib
                    norm_sum += norm_contrib
                    total_queries += query_count
                except Exception as e:
                    print(f"⚠️ Gradient estimation failed for one agent: {e}")

        if norm_sum > 0:
            x = x - step_size * (grad_sum / norm_sum)
        x = x.detach().clone().requires_grad_(True)
        # print("Updating State: ", step, x)

    return x, total_queries

def run_our_solution_concept_comparisons_parallel_sign_opt(x0, Sigma_set, lambda_mu_set, x_i_set, steps=1000, step_size=0.01, query_limit=10000):
    """
    Run iterative update using comparison-based gradient estimation.
    Parallelizes gradient estimation across agents.
    """
    # print("Query Limit: ", query_limit)
    x = x0.clone().detach().requires_grad_(True)
    total_queries = 0
    state_progression = [x.tolist()]
    for step in range(steps):
        grad_sum = torch.zeros_like(x)
        norm_sum = 0.0

        def estimate_for_agent(j):
            # print("Checking current state 1: ", x.shape)
            true_grad = compute_simplex_gradient(x, Sigma_set[j], lambda_mu_set[j])
            grad, query_count = gradient_estimation_sign_opt_batch(Sigma_set[j], lambda_mu_set[j], x, Q = query_limit)
            # print("checking gradient: ", angle_between(grad, true_grad))
            # print("Checking current state 2: ", x.shape)
            x_opt = x_i_set[j]
            diff_norm = torch.norm(x - x_opt)
            grad_unit = grad / torch.norm(grad)
            return grad_unit * diff_norm, diff_norm, query_count

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(estimate_for_agent, j) for j in range(len(lambda_mu_set))]
            for future in concurrent.futures.as_completed(futures):
                try:
                    grad_contrib, norm_contrib, query_count = future.result()
                    grad_sum += grad_contrib
                    norm_sum += norm_contrib
                    total_queries += query_count
                except Exception as e:
                    print(f"⚠️ Gradient estimation failed for one agent: {e}")

        if norm_sum > 0:
            x_new = x - step_size * (grad_sum / norm_sum)
            while ((x_new < 0).any()) and step_size > 1e-12:
                step_size *= 0.1
                x_new = x - step_size * (grad_sum / norm_sum)
            if step_size <= 1e-12:
                break
            state_progression.append(x_new.tolist())
            x = x_new.detach().clone().requires_grad_(True)
        # print("Updating State: ", step, x)

    return x, total_queries, state_progression