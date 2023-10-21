"""This module contains code for evaulation model metrics."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)

import wandb


class Metrics:
    """
    Contains methods for calculating performance metrics for model evaluation.
    """

    def __init__(self, cfg, confidence_threshold: float = 0.75):
        self.config = cfg
        self.confidence_threshold = confidence_threshold
        self.labels = self.config["labels"]

    @classmethod
    def from_dict(cls, cfg: dict):
        """
        Creates an instance of ClassName from a dictionary.
        Parameters:
        ----------
        cfg: dict
            A dictionary containing configuration information for the instance.

        Returns:
        -------
        ClassName
            An instance of the ClassName with the given configuration."""
        return cls(cfg, confidence_threshold=cfg.get("confidence_threshold"))

    def flat_accuracy(self, preds, labels):
        """
        Calculates the accuracy of the predictions by comparing them with the ground truth labels.
        Args:
            preds (numpy.ndarray): Array containing model predictions.
            labels (numpy.ndarray): Array containing ground truth labels.
        Returns:
            The accuracy of the predictions as a float.
        """
        pred_labels = (preds > self.confidence_threshold).astype(np.int32)
        pred_flat = pred_labels.flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def metric_fn(self, preds, labels):
        """
        Calculates precision, recall, f1-score, accuracy, and
        binary classification F1-score for the model predictions.
        Args:
            preds (list of numpy.ndarray): List containing the
            probability scores of the model predictions.
            labels (list of numpy.ndarray): List containing ground truth labels.
        Returns:
            A dictionary containing the calculated metrics as key-value pairs.
        """
        preds_flat = [i for a in preds for i in a]
        preds_flat = [1 if i >= self.confidence_threshold else 0 for i in preds_flat]
        true_labels_flat = [i for a in labels for i in a]

        precision, recall, f1__score, _ = precision_recall_fscore_support(
            true_labels_flat, preds_flat, average="binary"
        )
        f1score = f1_score(true_labels_flat, preds_flat)
        auc_ = accuracy_score(true_labels_flat, preds_flat)
        return {
            "eval_f1": f1__score,
            "eval_recall": recall,
            "eval_precision": precision,
            "auc": auc_,
            "f1score": f1score,
        }

    def print_metrics(self, result, mode):
        """
        Prints metrics evaluated and logs the information in wandb
        Args: Dictionary
        """
        recall = result["eval_recall"]
        precision = result["eval_precision"]
        f1score = result["f1score"]
        auc_ = result["auc"]
        print(f"{mode} Precision: {precision}")
        print(f"{mode} recall: {recall}")
        print(f"{mode} f1_score: {f1score}")
        print(f"{mode} auc: {auc_}")
        wandb.log({mode + "precision": precision})
        wandb.log({mode + "recall": recall})
        wandb.log({mode + "f1_score": f1score})
        wandb.log({mode + "auc": auc_})

    def roc_auc_confusion_matrix(self, preds, labels):
        """
        Plots the ROC curve and confusion matrix from the given predictions and labels
        Arg(s):
                preds(List): predictions of the model
                labels(List): True labels of the data
        """
        preds_flat = [i for a in preds for i in a.flatten()]
        labels_flat = [i for a in labels for i in a]
        df_plots = pd.DataFrame(
            data={"preds_flat": preds_flat, "labels_flat": labels_flat}
        )
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(labels_flat, preds_flat)
        roc_auc = auc(fpr, tpr)
        wandb.log({"roc_auc": auc(fpr, tpr)})
        # Create plot
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        image = wandb.Image(plt)
        wandb.log({"ROC curve": image})

        # compute confusion matrix
        binary_preds_flat = [
            1 if i >= self.config["confidence_threshold"] else 0 for i in preds_flat
        ]
        c_m = confusion_matrix(labels_flat, binary_preds_flat)
        # plot confusion matrix
        df_plots = pd.DataFrame(c_m, index=self.labels, columns=[self.labels])
        fig = plt.figure(figsize=(8, 8))
        a_x = fig.add_subplot(111)
        cax = a_x.matshow(df_plots)
        plt.title("Confusion matrix of the classifier")
        fig.colorbar(cax)
        a_x.set_xticklabels([""] + self.labels)
        a_x.set_yticklabels([""] + self.labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        image = wandb.Image(plt)
        wandb.log({"confusion_matrix": image})
