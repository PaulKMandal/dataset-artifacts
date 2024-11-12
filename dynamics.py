import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from scipy.stats import gaussian_kde
import random

def load_training_dynamics(td_dir):
    """
    Load training dynamics from a directory containing the 'training_dynamics.jsonl' file.
    Returns a dictionary mapping example indices to a list of records (one per epoch).
    """
    td_path = os.path.join(td_dir, 'training_dynamics.jsonl')
    dynamics = defaultdict(list)
    with open(td_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            idx = record['idx']
            dynamics[idx].append(record)
    return dynamics

def compute_metrics(dynamics):
    """
    Computes metrics for each example:
    - Average confidence in the true class.
    - Variability (standard deviation) of confidence in the true class.
    - Correctness (proportion of epochs where the prediction was correct).
    Returns a dictionary mapping example indices to a dictionary of metrics.
    """
    metrics = {}
    for idx, records in dynamics.items():
        correct = []
        confidences = []
        for record in records:
            # For NLI task
            if 'prob' in record:
                probs = np.array(record['prob'])
                pred_label = np.argmax(probs)
                true_label = record['label']
                is_correct = int(pred_label == true_label)
                confidence = probs[true_label]
            # For QA task
            elif 'start_prob' in record and 'end_prob' in record:
                start_probs = np.array(record['start_prob'])
                end_probs = np.array(record['end_prob'])
                pred_start = np.argmax(start_probs)
                pred_end = np.argmax(end_probs)
                true_start = record['start_position']
                true_end = record['end_position']
                is_correct = int(pred_start == true_start and pred_end == true_end)
                # Confidence as average of start and end probabilities of true positions
                confidence = (start_probs[true_start] + end_probs[true_end]) / 2.0
            else:
                continue  # Skip if data is missing
            correct.append(is_correct)
            confidences.append(confidence)
        if confidences:
            # Compute metrics
            avg_confidence = np.mean(confidences)
            variability = np.std(confidences)
            correctness = np.mean(correct)
            metrics[idx] = {
                'avg_confidence': avg_confidence,
                'variability': variability,
                'correctness': correctness
            }
    return metrics

def categorize_examples(metrics):
    """
    Categorizes examples into easy, ambiguous, and hard based on correctness.
    Returns a dictionary mapping example indices to categories.
    """
    categories = {}
    for idx, m in metrics.items():
        correctness = m['correctness']
        if correctness == 1.0:
            categories[idx] = 'Easy-to-learn'
        elif correctness >= 0.5:
            categories[idx] = 'Ambiguous'
        else:
            categories[idx] = 'Hard-to-learn'
    return categories

def plot_scatter(metrics, categories, output_dir, limit_scatter_samples=None, seed=None):
    """
    Plots a scatter plot of variability vs. average confidence.
    Points are colored based on the category.
    If limit_scatter_samples is set, randomly samples that number of examples for plotting.
    """
    # Prepare data for plotting
    indices = list(metrics.keys())

    # If limit_scatter_samples is specified, sample the indices
    if limit_scatter_samples is not None and len(indices) > limit_scatter_samples:
        if seed is not None:
            random.seed(seed)
        indices = random.sample(indices, limit_scatter_samples)

    x = []
    y = []
    colors = []
    category_colors = {
        'Easy-to-learn': 'green',
        'Ambiguous': 'orange',  # Changed from 'orange' to 'black'
        'Hard-to-learn': 'red'
    }
    for idx in indices:
        m = metrics[idx]
        x.append(m['variability'])
        y.append(m['avg_confidence'])
        colors.append(category_colors[categories[idx]])

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=colors, alpha=.7, s=1)  # Dots are smaller, solid color
    plt.xlabel('Variability (Std Dev of Confidence)')
    plt.ylabel('Average Confidence in True Class')
    plt.title('Confidence vs. Variability Scatter Plot')
    plt.grid(True)

    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Easy-to-learn',
               markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Ambiguous',
               markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Hard-to-learn',
               markerfacecolor='red', markersize=8)
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_vs_variability_scatter.png'))
    plt.close()

def plot_histogram(data, xlabel, title, output_path):
    """
    Plots a density histogram of the given data.
    """
    plt.figure(figsize=(8, 5))
    density = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 200)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    plt.plot(xs, density(xs))
    plt.fill_between(xs, density(xs), alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot Training Dynamics Analysis')
    parser.add_argument('--td_dir', type=str, required=True,
                        help='Directory containing training_dynamics.jsonl')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save plots and outputs')
    parser.add_argument('--limit_scatter_samples', type=int, default=None,
                        help='Limit the number of samples in the scatter plot by random sampling.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for sampling and reproducibility.')
    args = parser.parse_args()

    # Load training dynamics
    print("Loading training dynamics...")
    dynamics = load_training_dynamics(args.td_dir)

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(dynamics)

    # Categorize examples
    print("Categorizing examples...")
    categories = categorize_examples(metrics)

    # Plot scatter plot
    print("Plotting confidence vs. variability scatter plot...")
    plot_scatter(metrics, categories, args.output_dir,
                 limit_scatter_samples=args.limit_scatter_samples,
                 seed=args.seed)

    # Prepare data for histograms
    avg_confidences = [m['avg_confidence'] for m in metrics.values()]
    variabilities = [m['variability'] for m in metrics.values()]
    correctnesses = [m['correctness'] for m in metrics.values()]

    # Plot histograms
    print("Plotting histograms...")
    plot_histogram(avg_confidences, 'Average Confidence in True Class',
                   'Density vs. Confidence',
                   os.path.join(args.output_dir, 'density_vs_confidence.png'))
    plot_histogram(variabilities, 'Variability (Std Dev of Confidence)',
                   'Density vs. Variability',
                   os.path.join(args.output_dir, 'density_vs_variability.png'))
    plot_histogram(correctnesses, 'Correctness (Proportion Correct Predictions)',
                   'Density vs. Correctness',
                   os.path.join(args.output_dir, 'density_vs_correctness.png'))

    # Save categorized examples
    print("Saving categorized examples...")
    with open(os.path.join(args.output_dir, 'categorized_examples.json'), 'w') as f:
        json.dump(categories, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
