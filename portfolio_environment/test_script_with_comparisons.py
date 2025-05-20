import torch
import random
import json
import argparse
import numpy as np
import concurrent.futures
from helper_functions import sample_from_simplex, sample_investor_windows_and_lambdas, \
    setup_markowitz_environment_cached
from solution_concepts import solve_markowitz, run_our_solution_concept_actual, \
    run_our_solution_concept_comparisons_parallel_sign_opt


def single_test_run(num_agents, n, seed_offset=0, query_limit=10000):
    """Runs a single test of portfolio optimization and comparison-based bargaining.

    This function samples a set of agent-specific market environments using historical
    data, solves the Markowitz optimization for each agent to obtain optimal portfolios,
    and then runs both the actual and comparison-based solution concepts starting from a
    common initial state. It retries sampling if any optimization fails.

    Args:
        num_agents (int): Number of agents participating in the portfolio optimization.
        n (int): Number of assets (stocks) to include in the portfolio.
        seed_offset (int, optional): Offset added to the base seed for reproducibility. Defaults to 0.
        query_limit (int, optional): Maximum number of allowed queries for the comparison-based algorithm. Defaults to 10000.

    Returns:
        tuple: A 5-tuple containing:
            - final_simplex (list): Final weights computed by the actual solution concept.
            - final_simplex_comparison (list): Final weights from the comparison-based method.
            - starting_state_w (list): Initial state (uniform simplex sample).
            - query_count_ours (int): Number of queries used in the comparison-based method.
            - solution_set_list (list of lists): Optimal weights for each agent from Markowitz solutions.
    """
    base_seed = 42 + seed_offset
    torch.manual_seed(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)

    with open('top_100_tickers_2023.json', 'r') as f:
        tickers = json.load(f)[:n]

    success = False
    attempt = 0
    solution_set = []
    solution_set_list = []
    Sigma_set_list = []
    lambda_mu_set_list = []
    while not success:
        # Increment seed to ensure variation across retries
        seed = base_seed + attempt
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        start_date_list, end_date_list, lambda_vals = sample_investor_windows_and_lambdas(num_agents)
        Sigma_set = []
        lambda_mu_set = []
        Sigma_set_list = []
        lambda_mu_set_list = []
        for agent in range(num_agents):
            Sigma, lambda_mu, _ = setup_markowitz_environment_cached(
                tickers, start_date_list[agent], end_date_list[agent], lambda_vals[agent])
            Sigma_set_list.append(Sigma.tolist())
            lambda_mu_set_list.append(lambda_mu.tolist())
            Sigma_set.append(torch.tensor(Sigma, dtype=torch.float64))
            lambda_mu_set.append(torch.tensor(lambda_mu, dtype=torch.float64))

        solution_set = []
        solution_set_list = []
        valid = True
        for Sigma, lambda_mu in zip(Sigma_set, lambda_mu_set):
            w_opt = solve_markowitz(Sigma, lambda_mu)
            if w_opt is None:
                valid = False
                break
            solution_set.append(w_opt)
            solution_set_list.append(w_opt.detach().cpu().numpy().tolist())

        if valid:
            success = True
        else:
            print(f"Resampling due to solver failure... (seed: {seed})")
            attempt += 1

    starting_state_w = torch.tensor(sample_from_simplex(n), dtype=torch.float64)
    final_point_comparisons, query_count_ours, state_progression_ours_comparisons = run_our_solution_concept_comparisons_parallel_sign_opt(
        starting_state_w, Sigma_set, lambda_mu_set, solution_set, query_limit=query_limit)
    final_point, state_progression_ours = run_our_solution_concept_actual(starting_state_w, Sigma_set, lambda_mu_set,
                                                                          solution_set)

    final_simplex = final_point
    final_simplex_comparison = final_point_comparisons

    return final_simplex.tolist(), final_simplex_comparison.tolist(), starting_state_w.tolist(), query_count_ours, solution_set_list


if __name__ == "__main__":
    # Run a batch of single_test_run simulations across multiple agent counts and asset sizes.
    # Results are saved in JSON format for later analysis. Comparison query limit is configurable via CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_limit", type=int, default=1000,
                        help="Maximum number of queries allowed for comparison-based method")
    args = parser.parse_args()
    seed = 42
    torch.set_default_dtype(torch.float64)
    num_agents_list = [2, 3, 5, 10]
    n_list = [5, 10, 20, 50]
    distance_dict = {}
    num_tests = 100
    query_limit = args.comparison_limit

    for num_agents in num_agents_list:
        distance_dict[num_agents] = {}
        for n in n_list:
            print(
                f"Running {num_tests} tests for {num_agents} agents and {n} stocks with query limit {query_limit} ...")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(single_test_run, num_agents, n, i, query_limit=query_limit) for i in
                           range(num_tests)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            distance_dict[num_agents][n] = results
        with open(f'portfolio_comparisons_results_{query_limit}.json', 'w') as f:
            json.dump(distance_dict, f)