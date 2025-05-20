import torch

def query(Sigma, lambda_mu, current_state, offer):
    """
    Comparison query between current state and offered perturbation, using barrier penalties.
    """
    new_state = current_state + offer
    w_current = current_state
    w_next = new_state

    current_val = torch.dot(w_current, Sigma @ w_current) - torch.dot(lambda_mu, w_current)
    next_val = torch.dot(w_next, Sigma @ w_next) - torch.dot(lambda_mu, w_next)

    current_total = current_val
    next_total = next_val
    return next_total > current_total

def barrier_batch(w_batch, v_batch):
    Q = w_batch.shape[0]
    device = w_batch.device
    dtype = w_batch.dtype  # should be torch.float64

    # Force penalties to be float64
    penalties = torch.full((Q,), 1e6, dtype=dtype, device=device)

    # Guard against float32 eps or float literals
    eps = torch.tensor(1e-6, dtype=dtype, device=device)
    one = torch.tensor(1.0, dtype=dtype, device=device)

    valid_mask = (w_batch > 0).all(dim=1) & (v_batch.sum(dim=1) < 1.0)

    if valid_mask.any():
        w_valid = w_batch[valid_mask]
        v_valid = v_batch[valid_mask]

        penalty_values = -torch.sum(torch.log(w_valid + eps), dim=1) \
                         - torch.log(one - v_valid.sum(dim=1) + eps)

        # force dtype just to be absolutely sure
        penalty_values = penalty_values.to(dtype=dtype)

        penalties[valid_mask] = penalty_values

    return penalties

def query_batch(Sigma, lambda_mu, current_state, offer_batch):
    """
    Batched version of the comparison query function.

    Args:
        Sigma (torch.Tensor): Covariance matrix.
        lambda_mu (torch.Tensor): Mean-return tradeoff vector.
        current_state (torch.Tensor): Current point (1D tensor).
        offer_batch (torch.Tensor): Batch of offers, shape [Q, d].

    Returns:
        torch.BoolTensor: Comparison results for each offer, shape [Q].
    """
    Q, d = offer_batch.shape
    device = offer_batch.device
    new_states = current_state.unsqueeze(0) + offer_batch  # [Q, d]

    # Transform current_state (single vector) to simplex once
    w_current_batch = current_state.unsqueeze(0).expand(Q, -1)  # [d+1]

    # Transform new_states (batch of vectors) to simplex
    w_next_batch = new_states  # [Q, d+1]

    # Objective values
    def objective(w_batch):
        # print("Checking Objective: ", w_batch.shape, Sigma.shape, lambda_mu.shape)
        quad = torch.einsum('bi,ij,bj->b', w_batch, Sigma, w_batch)
        linear = w_batch @ lambda_mu  # <--- FIXED
        return quad - linear
    current_val = objective(w_current_batch)
    next_val = objective(w_next_batch)

    current_total = current_val
    next_total = next_val

    # current_total = current_val + barrier_coeff * current_barrier
    # next_total = next_val + barrier_coeff * next_barrier
    return next_total > current_total  # [Q]


def generate_offers(center_of_cone: torch.Tensor, step_size_orth=0.001):
    """
    Generate orthogonal perturbations around the cone center.
    """
    d = center_of_cone.shape[0]
    center = center_of_cone / torch.norm(center_of_cone)
    Q = torch.eye(d)
    Q[:, 0] = center
    Q, _ = torch.linalg.qr(Q)
    offers = Q[:, 1:].T * step_size_orth
    return [v for v in offers]

def refine_cone(center_of_cone, theta, offers, offer_responses):
    """
    Refine cone center using polar updates depending on offer outcomes.
    """
    sum_value = center_of_cone / torch.norm(center_of_cone)
    for i in range(len(offer_responses)):
        direction = offers[i] / torch.norm(offers[i])
        sign = 1.0 if offer_responses[i] else -1.0
        w_i = center_of_cone * torch.cos(theta) + sign * direction * torch.sin(theta)
        sum_value += w_i

    new_center = sum_value / torch.norm(sum_value)
    scaling_factor = torch.sqrt(torch.tensor((2 * len(center_of_cone) - 1) / (2 * len(center_of_cone))))
    new_theta = torch.arcsin(scaling_factor * torch.sin(theta))
    return new_center, new_theta

def gradient_estimation_sign_opt(Sigma, lambda_mu, current_state, Q=50, epsilon=1e-3):
    """
    Estimate the gradient of the objective using Sign-OPT with overshoot correction.

    Args:
        Sigma (torch.Tensor): Covariance matrix.
        lambda_mu (torch.Tensor): Mean-return tradeoff vector.
        current_state (torch.Tensor): Current point in the unconstrained space.
        Q (int): Number of random directions to sample.
        epsilon (float): Initial perturbation size.

    Returns:
        torch.Tensor: Estimated gradient direction (normalized).
    """
    estimated_grad = torch.zeros_like(current_state)

    for _ in range(Q):
        u = torch.randn_like(current_state)
        u = u / torch.norm(u)  # Normalize direction
        epsilon_temp = epsilon

        # Try to avoid ambiguous query results
        while epsilon_temp >= 1e-20:
            # print("Check: ", len(current_state))
            response = query(Sigma, lambda_mu, current_state, epsilon_temp * u)
            neg_response = query(Sigma, lambda_mu, current_state, -1.0 * epsilon_temp * u)

            # Break when responses differ (meaning there's preference signal)
            if response != neg_response:
                break

            epsilon_temp *= 0.1  # Reduce step if response is ambiguous (overshooting)

        # If still ambiguous after reduction, skip this sample
        if response == neg_response:
            continue

        sign = 1.0 if response else -1.0
        estimated_grad += sign * u

    # Avoid zero vector in rare degenerate cases
    if torch.norm(estimated_grad) < 1e-12:
        return torch.randn_like(current_state)  # fallback direction

    return estimated_grad / torch.norm(estimated_grad)

def gradient_estimation_sign_opt_batch(Sigma, lambda_mu, current_state, Q=10000, epsilon=1e-3, max_attempts=10):
    """
    Estimate the gradient of the objective using Sign-OPT with batched queries and per-direction overshoot correction.
    """
    # print("Query Limit Internal: ", Q)
    device = current_state.device
    d = current_state.shape[0]

    # Sample and normalize Q random directions (float64-safe)
    directions = torch.randn(Q, d, dtype=torch.float64, device=device)
    directions = directions - directions.mean(dim=1, keepdim=True)  # Project to tangent space
    directions = directions / directions.norm(dim=1, keepdim=True)  # Re-normalize

    epsilon_vec = torch.full((Q,), epsilon, dtype=torch.float64, device=device)
    accepted_mask = torch.zeros(Q, dtype=torch.bool, device=device)
    signs = torch.zeros(Q, dtype=torch.float64, device=device)

    for _ in range(max_attempts):
        pos_perturb = directions * epsilon_vec.unsqueeze(1)
        neg_perturb = -directions * epsilon_vec.unsqueeze(1)

        response_pos = query_batch(Sigma, lambda_mu, current_state, pos_perturb)
        response_neg = query_batch(Sigma, lambda_mu, current_state, neg_perturb)

        clear_mask = (response_pos != response_neg) & (~accepted_mask)

        pos_one = torch.tensor(1.0, dtype=signs.dtype, device=signs.device)
        neg_one = torch.tensor(-1.0, dtype=signs.dtype, device=signs.device)
        signs[clear_mask] = torch.where(response_pos[clear_mask], pos_one, neg_one)

        accepted_mask |= clear_mask
        ambiguous_mask = (response_pos == response_neg) & (~accepted_mask)
        epsilon_vec[ambiguous_mask] *= 0.1

        if accepted_mask.all():
            break

    if accepted_mask.sum() == 0:
        return torch.randn(current_state.shape, dtype=torch.float64, device=device)

    weighted_dirs = directions[accepted_mask] * signs[accepted_mask].unsqueeze(1)
    estimated_grad = weighted_dirs.sum(dim=0)

    return estimated_grad / estimated_grad.norm(), Q

def estimate_gradient_f_i_comparisons(x, Sigma, lambda_mu, theta_threshold=0.001):
    """
    Estimate gradient via cone refinement using binary comparison oracle.
    """
    num_dim = x.shape[0]
    cone_center = torch.zeros(num_dim)
    query_count = 0
    step_size_init = 0.001

    for index in range(num_dim):
        init_offer = torch.zeros(num_dim)
        init_offer[index] = step_size_init
        response = query(Sigma, lambda_mu, x, init_offer)
        neg_response = query(Sigma, lambda_mu, x, -1 * init_offer)
        while response == neg_response and step_size_init > 1e-20:
            step_size_init *= 0.1
            init_offer[index] = step_size_init
            response = query(Sigma, lambda_mu, x, init_offer)
            neg_response = query(Sigma, lambda_mu, x, -1 * init_offer)
            query_count += 2
        cone_center[index] = 1.0 if response else -1.0

    cone_center = cone_center / torch.norm(cone_center)
    theta = torch.acos(torch.tensor(1.0) / torch.sqrt(torch.tensor(float(num_dim))))

    while theta > theta_threshold:
        offers = generate_offers(cone_center)
        responses = []
        for offer in offers:
            response = query(Sigma, lambda_mu, x, offer)
            neg_response = query(Sigma, lambda_mu, x, -1 * offer)
            query_count += 2
            scale_down = 1e-1
            while response == neg_response and scale_down > 1e-5:
                scaled_offer = scale_down * offer
                response = query(Sigma, lambda_mu, x, scaled_offer)
                neg_response = query(Sigma, lambda_mu, x, -1 * scaled_offer)
                query_count += 2
                scale_down *= 0.1
            responses.append(response)
        cone_center, theta = refine_cone(cone_center, theta, offers, responses)

    return cone_center, query_count

def random_spd_matrix(n, scale=1.0):
    A = torch.randn(n, n)
    return scale * (A @ A.T) + torch.eye(n) * 1e-3  # Ensure positive definiteness

def sample_random_simplex(n):
    x = torch.rand(n)
    return x / x.sum()