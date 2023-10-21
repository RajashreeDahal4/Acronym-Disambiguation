"""This module is used for validation of dataset."""
import torch
from metrics_evaluation import Metrics
import wandb


class Validator:
    """Creates a validation model to evaluate the accuracy of a given model on
    a validation dataset."""

    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = torch.device(device)
        self.model = None

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
            An instance of the ClassName with the given configuration.
        """
        return cls(cfg, device=cfg.get("device"))

    def loop_dataloader(self, dataloader, mode, model):
        """Loop over the batches in a PyTorch dataloader and
        compute predictions using a PyTorch model.
        Args:
            dataloader (torch.utils.data.DataLoader):
            A PyTorch dataloader containing the input data.
        mode (str): A string specifying the mode of operation("pred", "test", or "validation").
        model (torch.nn.Module): A PyTorch model used to generate predictions.
        Returns:
            Tuple of the form (accuracy, nb_steps, predictions, labels):
                - accuracy (float): The average accuracy of the predictions over all batches.
                - nb_steps (int): The number of batches in the dataloader.
                - predictions (list): A list of NumPy arrays containing
                the predicted labels for each batch.
                - labels (list): A list of NumPy arrays containing the true labels for each batch.
        """
        # Validation
        # Tracking variables
        metrics_dict = {"predictions": [], "labels": [], "accuracy": 0, "nb_steps": 0}
        metrics = Metrics.from_dict(self.config)
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                logits = model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )
            logits = logits["logits"]

            # Applying sigmoid to the  element of the tensor
            logits = torch.sigmoid(logits)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            if mode in ["test", "validation"]:
                tmp_accuracy = metrics.flat_accuracy(logits, label_ids)
                metrics_dict["accuracy"] += tmp_accuracy
                metrics_dict["nb_steps"] += 1
                wandb.log(
                    {
                        mode
                        + " Accuracy": metrics_dict["accuracy"]
                        / metrics_dict["nb_steps"]
                    }
                )
            # Store predictions and true labels
            metrics_dict["predictions"].append(logits)
            metrics_dict["labels"].append(label_ids)
        return (
            metrics_dict["accuracy"],
            metrics_dict["nb_steps"],
            metrics_dict["predictions"],
            metrics_dict["labels"],
        )

    def model_validation(self, validation_dataloader, model):
        """
        Calculate the validation accuracy and return the predictions and labels.

        Args:
            validation_dataloader (DataLoader): The PyTorch data loader for the validation set.
            model1 (torch.nn.Module): The PyTorch model to evaluate.

        Returns:
            A tuple of two lists: the predicted probabilities for each
            sample in the validation set, and the corresponding
            true labels.
        """
        self.model = model
        self.model.eval()
        eval_accuracy, nb_eval_steps, predictions, labels = self.loop_dataloader(
            validation_dataloader, mode="validation", model=model
        )
        print(f"Validation Accuracy: {eval_accuracy / nb_eval_steps}")
        wandb.log({"Epoch Validation Accuracy": (eval_accuracy / nb_eval_steps)})
        return predictions, labels
