""" Importing modules for training the model"""
import torch
from tqdm import trange
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.utils import logging
from model import ModelBert
from validation import Validator
import wandb

logging.set_verbosity(40)


class Trainer:
    """
    A class for training a BERT model on a given dataset.
    """

    def __init__(self, config, epochs: int = 2):
        self.config = config
        self.epochs = epochs
        model = ModelBert.from_dict(self.config)
        self.model1, self.tokenizer = model.make_model()
        self.model1.resize_token_embeddings(len(self.tokenizer))
        self.optimizer_grouped_parameters1 = [
            # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
            {
                "params": [
                    p
                    for n, p in list(self.model1.named_parameters())
                    if not any(nd in n for nd in self.config["no_decay"])
                ],
                "weight_decay_rate": 0.1,
            },
            # Filter for parameters which *do* include those.
            {
                "params": [
                    p
                    for n, p in list(self.model1.named_parameters())
                    if any(nd in n for nd in self.config["no_decay"])
                ],
                "weight_decay_rate": 0.0,
            },
        ]
        self.optimizer1 = AdamW(
            self.optimizer_grouped_parameters1,
            lr=self.config["learning_rate"],
            eps=self.config["eps"],
        )
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
            An instance of the ClassName with the given configuration."""
        return cls(cfg, epochs=cfg.get("epochs"))

    def model_train(self, train_dataloader, validation_dataloader):
        """
        Trains  BERT model on the given training data and validates it on the validation data.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The data loader for training data.
            validation_dataloader (torch.utils.data.DataLoader):
            The data loader for validation data.

        Returns:
            Tuple of lists: The predictions and labels for the validation data.
        """

        # Separate the `weight` parameters from the `bias` parameters.
        # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
        # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
        # Create the learning rate scheduler.
        scheduler1 = get_linear_schedule_with_warmup(
            self.optimizer1,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * self.epochs,
        )
        for _, batch in enumerate(train_dataloader):
            batch = tuple(t.to(self.config["device"]) for t in batch)
        self.model = self.model1

        # trange is a tqdm wrapper around the normal python range
        for _ in trange(self.epochs, desc="Epoch"):
            self.model.train()
            # Tracking variables
            train_loss = 0
            num_train_steps = 0
            # Train the data for one epoch
            for _, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to(torch.device(self.config["device"])) for t in batch)
                # Unpack the inputs from our dataloader
                # Clear out the gradients (by default they accumulate)
                self.optimizer1.zero_grad()
                # Forward pass
                outputs = self.model(
                    batch[0],
                    token_type_ids=None,
                    attention_mask=batch[1],
                    labels=batch[2],
                )
                logits = outputs["logits"]
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(logits.view(-1), batch[2].float())
                wandb.log({"Training Loss": loss.item()})
                # Backward pass
                loss.mean().backward()
                # Update parameters and take a step using the computed gradient
                self.optimizer1.step()
                # Update the learning rate.
                scheduler1.step()
                # Update tracking variables
                train_loss += loss.item()
                num_train_steps += 1
                print("the training step is", num_train_steps)
            train_loss = train_loss / num_train_steps
            print("Train loss: ", train_loss)
            wandb.log({"Epoch Training Loss": (train_loss / num_train_steps)})
            # validation
            validation = Validator.from_dict(self.config)
            predictions, labels = validation.model_validation(
                validation_dataloader, self.model
            )
        torch.save(self.model.state_dict(), self.config["saved_model_path"])
        return predictions, labels
