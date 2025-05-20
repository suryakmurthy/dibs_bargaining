# Direction-Based Bargaining (DiBS)

This repository contains code for reproducing the experiments in:

**"Cooperative Bargaining Without Utilities: Mediated Solutions from Direction Oracles"**  

DiBS is a bargaining algorithm that uses only agents’ most preferred directions (direction oracles), offering solutions that are robust to nonaffine utility transformations and do not require explicit utility values.

We include two experiments:
- **Robot Formation (Multi-Agent Assignment)**
- **Mediated Portfolio Management (Financial Resource Allocation)**

## Dependencies

This project requires **Python 3.9.21** and the following key Python packages:

- `numpy`, `scipy`, `pandas` — numerical computing and data manipulation
- `matplotlib`, `seaborn`, `tikzplotlib` — plotting and figure generation
- `cvxpy` — convex optimization
- `yfinance` — used to retrieve historical stock data
- `torch` — used for implementing comparison-based gradient estimation

To install all required dependencies, use the provided `requirements.txt`.
To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Repository Structure

This codebase is organized around two main experimental domains:

### `portfolio_environment/`
Implements the **mediated portfolio management** experiment (Section 4.2 in the paper).

- `test_script_with_comparisons.py` — Runs the DiBS algorithm in the portfolio setting and generates JSON result files.
- `data_parser_comparisons.py` — Parses the generated results and creates the plots shown in the paper.

### `robot_environment/`
Implements the **multi-agent formation assignment** experiment (Section 4.1 in the paper).

- `main.py` — Runs the baseline formation control experiment with standard agent preferences.
- `main_squared.py` — Runs the same experiment but with **squared utilities for one group**, simulating a nonaffine transformation.

## Running the Experiments

Each experiment can be run independently by executing the corresponding script in its folder. All output figures and result files will be saved locally.

---
### Multi-Agent Formation Assignment

Navigate to the `robot_environment/` directory and run one of the following:

```bash
# Standard utility setting
python main.py

# Squared utility transformation for one group
python main_squared.py
```

Each script will produce trajectory plots showing how agent positions evolve under DiBS, NBS, and KSBS.

- **Output**:  
  - Plots from `main.py` will be saved in the `plots/` directory.  
  - Plots from `main_squared.py` will be saved in the `plots_squared/` directory.

---

### Mediated Portfolio Management

To reproduce the results from the portfolio allocation experiment (Section 4.2), navigate to the `portfolio_environment/` directory.

#### Step 1: Run the experiment

```bash
python test_script_with_comparisons.py --comparison_limit 1000
```

**Arguments:**
- `--comparison_limit`: Number of comparison queries per agent per iteration (e.g., 1, 10, 100, 1000, 10000).

This script runs DiBS with direction oracles estimated via comparisons, and saves the results in a JSON file:

```
portfolio_comparisons_results_<comparison_limit>.json
```

#### Step 2: Generate plots

Once the data has been collected, run the following script to generate the accuracy plots:

```bash
python data_parser_comparisons.py --num_agents 10
```

**Arguments:**
- `--num_agents`: The number of agents used in the experiments (must match the value used during data generation).

This script will load the corresponding JSON files (e.g., `portfolio_comparisons_results_<comparison_limit>.json`) and produce box plots showing the relative error of DiBS across different comparison budgets and stock counts.

- **Output**:  
  - All generated box plots will be saved in the `plots/` directory.
