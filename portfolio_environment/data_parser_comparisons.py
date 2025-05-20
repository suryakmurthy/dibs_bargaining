import json
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

def euclidean_distance(a, b):
    """Computes the Euclidean distance between two vectors.

    Args:
        a (array-like): First vector.
        b (array-like): Second vector.

    Returns:
        float: Euclidean distance between vectors a and b.
    """
    return np.linalg.norm(np.array(a) - np.array(b))

def main():
    """Main entry point for generating and plotting relative distance errors.

    Loads precomputed JSON results of portfolio optimization for different numbers of
    comparisons and assets. Computes the relative distance between the solutions produced
    by the actual and comparison-based methods. Outputs summary statistics and generates
    a box plot (in both TikZ and PDF format) showing relative distances on a log scale.

    Command-line Args:
        --num_agents (str): The number of agents used in the experiment. Must match a key
                            in the result files (as a string).

    Outputs:
        - Console summary of mean and standard deviation of distances per configuration.
        - TikZ and PDF plots saved under 'plots/' showing relative distances per setting.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=str, required=True, help="Number of agents")
    args = parser.parse_args()
    num_agents = args.num_agents

    suffixes = ['1', '10', '100', '1000', '10000']
    num_stocks = ["5", "10", "20", "50"]
    all_distances = []

    for suffix in suffixes:
        file_path = f"portfolio_comparisons_results_{suffix}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        for s in num_stocks:
            if num_agents not in data or s not in data[num_agents]:
                continue

            distance_ours_comparisons = []
            for entry in data[num_agents][s]:
                (
                    final_ours, final_comp, starting_state_w, query_count_ours, solution_set_list
                ) = entry

                dist = euclidean_distance(final_ours, final_comp) / euclidean_distance(final_ours, starting_state_w)
                distance_ours_comparisons.append(dist)

            print(f"Results with {num_agents} agents and {s} stocks")
            print(f"→ Mean Relative Distance Error: {np.mean(distance_ours_comparisons):.10f}")
            print(f"→ Std Dev: {np.std(distance_ours_comparisons):.10f}")
            print(f"→ Trials: {len(distance_ours_comparisons)}")
            print("-" * 60)

            for d in distance_ours_comparisons:
                all_distances.append({
                    "Stocks": int(s),
                    "Comparisons": int(suffix),
                    "Distance": d
                })

    df = pd.DataFrame(all_distances)
    palette = sns.color_palette("pastel")[:len(df["Stocks"].unique())]
    df["Comparisons"] = pd.to_numeric(df["Comparisons"])

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=df,
        x="Comparisons",
        y="Distance",
        hue="Stocks",
        palette=palette,
        dodge=True,
        width=0.6,
        showcaps=True,
        boxprops={"edgecolor": "gray", "linewidth": 1.2},
        whiskerprops={"color": "gray", "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        medianprops={"color": "black", "linewidth": 1.5},
        ax=ax
    )

    ax.set_yscale("log")
    ax.set_xlabel("Number of Comparisons (log scale)")
    ax.set_ylabel("Relative Distance Between Solutions (log scale)")
    ax.set_title(f"Distance Between Our and Comparison-Based Solutions\n({num_agents} Agents)")
    ax.get_legend().remove()

    tikzplotlib.save(f"plots/box_plot_{num_agents}_agents.tex")
    plt.savefig(f"plots/box_plot_{num_agents}_agents.pdf")
    plt.show()

if __name__ == "__main__":
    main()