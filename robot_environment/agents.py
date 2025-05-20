import torch
import numpy as np

class AgentUtilityFunction:
    def __init__(self, agent_id, other_agent_ids, alphas, betas,
                 group_assignments,
                 same_group_weight=1.0,
                 scaling_factor=1.0,
                 nonlinear_transform=None,
                 device="cpu", dtype=torch.float32,
                 center_gamma=None, center_point=(5.0, 5.0)):
        """
        Initializes the utility function for a single agent.

        Args:
            nonlinear_transform (str or None): Type of non-linear transform to apply.
                Options: None, "square".
        """
        self.agent_id = agent_id
        self.other_agent_ids = other_agent_ids
        self.device = device
        self.dtype = dtype

        self.alphas = {k: torch.tensor(v, device=device, dtype=dtype) for k, v in alphas.items()}
        self.betas = {k: torch.tensor(v, device=device, dtype=dtype) for k, v in betas.items()}
        self.scaling_factor = torch.tensor(scaling_factor, device=device, dtype=dtype)

        self.group_assignments = group_assignments
        self.same_group_weight = same_group_weight

        self.center_gamma = torch.tensor(center_gamma, device=device, dtype=dtype) if center_gamma is not None else None
        self.center_point = torch.tensor(center_point, device=device, dtype=dtype)

        self.nonlinear_transform = nonlinear_transform  # <-- NEW

        assert set(self.other_agent_ids) == set(self.alphas.keys()) == set(self.betas.keys()), \
            "Mismatch between other_agent_ids, alphas, and betas keys."

    def compute_utility_distances(self, distances):
        """
        Computes utility from a list of distances:
        - First len(other_agent_ids) entries: distances to other agents
        - Final entry: distance to center
        """
        assert len(distances) == len(self.other_agent_ids) + 1, \
            f"Expected {len(self.other_agent_ids) + 1} distances, got {len(distances)}"

        utility = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        for idx, j in enumerate(self.other_agent_ids):
            d_ij = distances[idx]
            alpha_ij = self.alphas[j]
            beta_ij = self.betas[j]

            if self.group_assignments[self.agent_id] == self.group_assignments[j]:
                weight = self.same_group_weight
            else:
                weight = 1.0

            interaction = torch.exp(-alpha_ij * d_ij) - torch.exp(-beta_ij * d_ij)
            utility += weight * interaction

        # Center attraction from the last entry
        if self.center_gamma is not None:
            d_center = distances[-1]
            utility += 10 * torch.exp(-self.center_gamma * d_center)

        # Apply non-linear transform if requested
        if self.nonlinear_transform is not None:
            if self.nonlinear_transform == "square":
                utility = (utility ** 2) * utility/abs(utility)
            else:
                raise ValueError(f"Unknown nonlinear transform: {self.nonlinear_transform}")

        return self.scaling_factor * utility

    def compute_utility(self, positions):
        """
        Computes the utility of this agent based on current positions.
        """
        p_i = positions[self.agent_id]
        utility = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        for j in self.other_agent_ids:
            p_j = positions[j]
            d_ij = torch.norm(p_i - p_j)
            alpha_ij = self.alphas[j]
            beta_ij = self.betas[j]

            # Determine weight
            if self.group_assignments[self.agent_id] == self.group_assignments[j]:
                weight = self.same_group_weight
            else:
                weight = 1.0

            # Weighted interaction
            interaction = (torch.exp(-alpha_ij * d_ij) - torch.exp(-beta_ij * d_ij))
            utility += weight * interaction

        # Add center attraction if enabled
        if self.center_gamma is not None:
            d_center = torch.norm(p_i - self.center_point)
            utility += 10 * torch.exp(-self.center_gamma * d_center)

        # Apply non-linear transform if requested
        if self.nonlinear_transform is not None:
            if self.nonlinear_transform == "square":
                utility = (utility ** 2) * utility/abs(utility)
            else:
                raise ValueError(f"Unknown nonlinear transform: {self.nonlinear_transform}")

        return self.scaling_factor * utility

    def compute_optimal_distance(self, j):
        """
        Analytically compute the ideal distance between two agents.
        """
        alpha = self.alphas[j]
        beta = self.betas[j]
        return torch.log(alpha / beta) / (alpha - beta)