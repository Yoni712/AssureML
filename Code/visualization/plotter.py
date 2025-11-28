# visualization/plotter.py

import matplotlib.pyplot as plt
import numpy as np

class Plotter:

    def plot_robustness_distribution(self, robustness_scores):
        plt.figure(figsize=(6,4))
        plt.hist(robustness_scores, bins=10, edgecolor='black')
        plt.title("Robustness Distribution")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.show()

    def plot_consistency_breakdown(self, scores_dict):
        """
        scores_dict example:
        {
            "brightness": 0.85,
            "rotation": 0.78,
            "noise": 0.91
        }
        """
        plt.figure(figsize=(6,4))
        keys = list(scores_dict.keys())
        values = list(scores_dict.values())

        plt.bar(keys, values)
        plt.ylim(0, 1)
        plt.title("Consistency Breakdown by Perturbation Type")
        plt.ylabel("Cosine similarity")
        plt.grid(axis='y', alpha=0.3)
        plt.show()

    def plot_variance_curve(self, variance_values):
        plt.figure(figsize=(6,4))
        plt.plot(variance_values, marker='o')
        plt.title("Variance Across Micro-noise Runs")
        plt.xlabel("Run")
        plt.ylabel("Variance")
        plt.grid(alpha=0.3)
        plt.show()
