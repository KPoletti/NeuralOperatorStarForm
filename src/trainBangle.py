"""
This module contains the Trainer class, which is used for training a neural network
model.
It also contains utility functions for checking the visibility of data and creating
animations.
"""

import logging
import os
import re
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from src.plottingUtils import createAnimation, Animation_true_pred_error

import torch
import wandb
from torch import device, nn
import torch.distributed as dist 
from neuralop import LpLoss, H1Loss
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.utils import count_model_params


sns.set_color_codes(palette="deep")
logger = logging.getLogger(__name__)
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
class NormalizedMSE:
    """Computes the normalized MSE loss for a given input tensors x and y.

    Args:
        object (_type_) : The object type.
    """

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=self.reduction)

    def __call__(self, x, y) -> float:
        """
        Args:
            x (torch.Tensor): The input tensor to compute the loss for.
            y (torch.Tensor): The target tensor to compute the loss for.
        Returns:
            float: The normalized MSE loss for the given input and target tensors.
        """
        # compute the loss
        return self.mse(x, y) / torch.mean(y)


def dataVisibleCheck(
    target: torch.Tensor,
    output: torch.Tensor,
    meta: dict,
    save: str,
    idx,
    device,
):
    """
    Creates a mp4 of a random batch of the target to check if the target is visible
    Inputs:
        target: pytorch tensor of shape (batch, time, x, y, variable)
        output: pytorch tensor of shape (batch, time, x, y, variable)
        meta: dictionary of metadata
        save: directory to save the plots
    """
    # check if FNO3d is in save
    if "intensity" in save:
        print(target.shape)
        target = target.sum(1)
        output = output.sum(1)
        # permute to get variable on the end
        target = target.permute(0, 3, 1, 2)
        output = output.permute(0, 3, 1, 2)
    elif "FNO3d" in save or "CNL2d" in save or "RNN" in save:
        # permute to get variable on the end
        target = target.permute(0, 4, 2, 3, 1)
        output = output.permute(0, 4, 2, 3, 1)

        target = target[..., 0]
        output = output[..., 0]
    a, b, c, d = target.shape
    target = target.reshape(a * b, c, d)
    output = output.reshape(a * b, c, d)
    mass = meta["mass"]
    time_data = torch.tensor(
        [[t[batch_id] for t in meta["time"]] for batch_id in range(a)]
    )
    time_data = time_data.flatten()
    if a <= 2:
        a = 2 * b
    Animation_true_pred_error(
        truthData=target[: a // 2, ...],
        predcData=output[: a // 2, ...],
        savePath=save,
        out_times=time_data[: a // 2],
        mass=mass,
        device=device,
        fps=10,
    )
    # createAnimation(target, time_data, save, mass, fps=1)


class TrainerDuo(object):
    """
    A class for training a neural network model.

    Attributes:
        model (nn.Module): The neural network model to train.
        params (dataclass): A dataclass containing the hyperparameters for training.
        device (torch.device): The device to use for training (e.g. "cpu" or "cuda").
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler
                                                           to use for training.
        loss (callable): The loss function to use for training.
        plot_path (str): The directory to save the plots.
    """

    def __init__(
        self,
        model: nn.Module,
        params: dataclass,
        device: torch.device,
        ckp_path: str = "",
        save_every: int = 1,
    ) -> None:
        """
        Initializes a Trainer object.

        Args:
            model (nn.Module): The neural network model to train.
            params (dataclass): A dataclass containing the hyperparameters for training.
            device (torch.device): The device to use for training (e.g. "cpu" or "cuda")
        """
        self.model = model
        self.params = params
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay,
            amsgrad=True,
        )
        if self.device == 0:
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            print(f"Adam LR {lr}")
        self.plot_path = f"results/{self.params.path}/plots"
        self.save_every = save_every
        self.epochs_run = 0
        self.ckp_path = ckp_path
        self.do_animate = (self.device != 0) * self.params.use_ddp
        print(self.do_animate)
        # initialize the loss function
        self.loss = self.params.loss_fn
        # calculate Learn-Rate warm up scheduler iterations
        ntrain = int((self.params.split[0] * self.params.N))
        warmup_ep = self.params.epochs * 5 // 100

        iterations = (
            (self.params.epochs - warmup_ep)
            * (ntrain // self.params.batch_size)
            // self.params.cosine_step
        )
        lin_iters = (
            warmup_ep * (ntrain // self.params.batch_size) // self.params.cosine_step
        )
        if self.params.use_ddp:
            self.num_gpus = torch.cuda.device_count()

        if os.path.exists(self.ckp_path) and params.use_ddp:
            self._load_snapshot(self.ckp_path)
        self.scheduler1 = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=lin_iters
        )
        self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=iterations,
            eta_min=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.scheduler1, self.scheduler2],
            milestones=[lin_iters],
        )
        # define a test loss function that will be the same regardless of training
        l2loss = LpLoss(d=self.params.d, p=2, reduce_dims=(0, 1))
        h1loss = H1Loss(d=self.params.d, reduce_dims=(0, 1))
        l2loss1d = LpLoss(d=1,p=2)
        def weighted_loss(x, y, mask, alpha=self.params.alpha):
            return (1 - alpha) * l2loss1d(x[mask], y[mask]) + alpha * l2loss1d(x[~mask], y[~mask])

 
        self.test_losses = {"l2loss": l2loss, "h1loss": h1loss, "weight": weighted_loss}

    def batchLoop(self, train_loader, output_encoder=None, epoch: int = 1):
        """
        A function to loop through the batches in the training data.

        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.
            output_encoder (callable): The output encoder to use for decoding the output
            epoch (int): The current epoch number.

        Returns:
            tuple: A tuple containing the training loss and the loss of the last batch.
        """
        # initialize loss
        loss = 0
        train_loss = 0
        # select a random batch
        for batch_idx, sample in enumerate(train_loader):
            self.optimizer.zero_grad(set_to_none=True)

            data = sample["x"].to(self.device)
            target = sample["y"].to(self.device)
            output = self.model(data)

            if epoch == 0 and batch_idx == 0:
                logger.debug(f"Batch Data Shape: {data.shape}")
                logger.debug(f"Batch Target Shape: {target.shape}")
                logger.debug(f"Output Data Shape: {output.shape}")
                logger.debug(f"Input Meta Data: {sample['meta_x']['mass'][0]}")
                logger.debug(f"Output Meta Data: {sample['meta_y']['mass'][0]}")
            # decode  if there is an output encoder
            if output_encoder is not None:
                # decode the target
                target = output_encoder.decode(target)
                # decode the output
                output = output_encoder.decode(output)
            mask = torch.stack([data[...,i] >= 3 * data[...,i].std() for i in range(data.shape[-1])],dim=-1)
            # compute the loss
            if self.params.loss_name == "weighted":
                loss = self.loss(output, target, mask)  
            else:
                # loss = self.loss(output, target)
                weights = torch.ones_like(output)
                weights[~mask] = self.params.alpha**(1/2)  
                loss = self.loss(weights*output, weights*target)
 
            # Backpropagate the loss
            loss.backward()
            self.optimizer.step()
            if hasattr(self, "scheduler"):
                self.scheduler.step()
            train_loss += loss.item()
            del output, target, data, loss
        torch.autograd.set_detect_anomaly(mode=False)
        return train_loss

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, self.ckp_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.ckp_path}")

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        output_encoder=None,
        sweep: bool = False,
    ) -> None:
        """
        Train the model
        Input:
            train_loader: torch.utils.data.DataLoader
            valid_loader: torch.utils.data.DataLoader
        Output:
            None
        """

        test_data_length = len(test_loader.dataset)
        train_data_length = len(train_loader.dataset)
        if self.params.use_ddp:
            test_data_length //= self.num_gpus
            train_data_length //= self.num_gpus
        if "RNN" in self.params.NN:
            test_data_length *= self.params.T
            train_data_length *= self.params.T

        logger.debug(f"len(test_loader.dataset) = {test_data_length}")
        logger.debug(f"len(train_loader.dataset) = {train_data_length}")
        old_test = 0
        early_stop = torch.zeros(1).to(self.device)

        # start training
        for epoch in range(self.epochs_run, self.params.epochs):
            epoch_timer = time.time()

            # set to train mode
            self.model.train()
            train_loss = self.batchLoop(train_loader, output_encoder, epoch)

            if (
                self.params.use_ddp
                and self.device == 0
                and epoch % self.save_every == 0
            ):
                self._save_snapshot(epoch)
            train_loss /= train_data_length

            self.model.eval()
            test_losses = {key: 0.0 for key in self.test_losses}
            for batch_idx, sample in enumerate(test_loader):
                data = sample["x"].to(self.device)
                target = sample["y"].to(self.device)

                output = self.model(data)
                # decode  if there is an output encoder
                if output_encoder is not None:
                    output = output_encoder.decode(output)
                    # do not decode the test target because test data isn't encoder
                mask = torch.stack([data[...,i] >= 3 * data[...,i].std() for i in range(data.shape[-1])],dim=-1)
               # mask = data > 3 * data.std() + data.mean()
                for key, loss_type in self.test_losses.items():
                    if key == "weight":
                        test_losses[key] += loss_type(output, target, mask).item()
                    else:
                        test_losses[key] += loss_type(output, target).item()
                del output, target, data
            for key in test_losses:
                test_losses[key] /= test_data_length

            # self.scheduler1.step()

            epoch_timer = time.time() - epoch_timer
            logger.info(
                "Epoch: %d/%d \t Took: %.2f \t Training Loss: %.6f \t Test L2 Loss: %.6f \t Test H1 Loss: %.6f \t Test Wt Loss: %.6f",
                epoch + 1,
                self.params.epochs,
                epoch_timer,
                train_loss,
                test_losses["l2loss"],
                test_losses["h1loss"],
                test_losses["weight"],
            )
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            if self.device == 0:
                print(f"Adam LR {lr}")

            print(
                f"Epoch: {epoch+1}/{self.params.epochs} \t"
                f"Took: {epoch_timer:.2f} \t"
                f"Training Loss: {train_loss:.6f} \t"
                f"Test L2 Loss: {test_losses['l2loss']:.6f} \t"
                f"Test H1 Loss: {test_losses['h1loss']:.6f} \t"
                f"Test Wt Loss: {test_losses['weight']:.6f} \t"
            )
            wandb.log(
                {
                    "Test-Loss-L2": test_losses["l2loss"],
                    "Test-Loss-H1": test_losses["h1loss"],
                    "Test-Loss-Wt": test_losses["weight"],
                    "Train-Loss": train_loss,
                    "Learn-Rate": lr,
                }
            )

            if (epoch >= 25) and np.abs(old_test - test_losses["weight"] ) <=1e-6:
                print("EARLY STOPPING CRITERIA MET ON EPOCH", epoch, "on device", device)
                early_stop += 1
            if self.params.use_ddp:
                dist.all_reduce(early_stop, op=dist.ReduceOp.SUM)
            if early_stop != 0:
                break
            old_test = test_losses["weight"]

        if self.params.saveNeuralNetwork:
            modelname = f"results/{self.params.path}/models/{self.params.NN}.pt"
            if self.params.use_ddp:
                torch.save(self.model.module.state_dict(), modelname)
            else:
                torch.save(self.model.state_dict(), modelname)

    def plot(
        self,
        prediction,
        in_data,
        truth,
        savename="",
        do_animate=True,
    ):
        """
        Plot the prediction
        Input:
            prediction: torch.tensor
            input: torch.tensor
            truth: torch.tensor

            savename: str
        Output:
            None
        """
        from scipy.stats import gaussian_kde

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        mask = torch.stack(
            [
                in_data[i] > 3 * in_data[i].std()
                for i in range(in_data.shape[0])
            ]
        )

        # flatten the masked data
        truth_flat = truth[mask].flatten()
        pred_flat = prediction[mask].flatten()
        # select a random set of indices to plot
        indices = np.random.choice(truth_flat.shape[0], 50000, replace=False)

        # create a density color gradient
        xy = np.vstack([truth_flat[indices], pred_flat[indices]])
        z = gaussian_kde(xy)(xy)
        angle = np.pi / 2
        fig_sc, ax = plt.subplots(1, 1)
        ax.scatter(
            truth_flat[indices],
            pred_flat[indices],
            c=z,
            # s=100,
            alpha=0.01,
        )
        ax.plot([-angle, angle], [-angle, angle], "r--")
        ax.set_xlabel("True Angle (radians)")
        ax.set_ylabel("FNO Predicted Angle (radians)")
        ax.set_ylim(truth_flat.min(),truth_flat.max())
        image_sc = wandb.Image(fig_sc)
        plt.savefig(f"{self.plot_path}/{savename}_scatter.png")
        plt.close()
        # convert to degrees
        truth_flat = truth_flat * 180 / np.pi
        pred_flat = pred_flat * 180 / np.pi

        # create a histogram of the data of the difference
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel(r"$\Phi_{FNO} - \Phi_{True}$ (degrees)")
        # annotate the plot with the mean and standard deviation
        ax.annotate(
            f"Mean ($\Phi_p - \Phi_T$): {torch.mean(pred_flat- truth_flat).item():0.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=12,
            ha="left",
            va="top",
        )
        ax.annotate(
            f"STD ($\Phi_p - \Phi_T$): {torch.std(pred_flat- truth_flat).item():0.3f}",
            xy=(0.05, 0.85),
            xycoords="axes fraction",
            fontsize=12,
            ha="left",
            va="top",
        )
        hist = ax.hist((pred_flat - truth_flat), bins=100, alpha=0.5, density=True)
        image_hist = wandb.Image(fig)
        plt.savefig(f"{self.plot_path}/{savename}_hist.png")
        wandb.log({"Scatter": image_sc, "Hist": image_hist})
        plt.close()

    def RMSE(self, prediction, truth, mask=None):
        if mask is not None:
            rmse = torch.sqrt(torch.mean((prediction[mask] - truth[mask]) ** 2))
            nrmse = rmse / torch.sqrt(torch.mean(truth[mask] ** 2))
            return rmse, nrmse
        rmse = torch.sqrt(torch.mean((prediction - truth) ** 2))
        nrmse = rmse / torch.sqrt(torch.mean(truth**2))
        return rmse, nrmse

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        input_encoder=None,
        output_encoder=None,
        savename: str = "",
        do_animate=True,
    ) -> None:
        """
        Evaluates the model on the given data_loader and saves the results to a .mat
        Args:
            data_loader (torch.utils.data.DataLoader): The data loader to evaluate the
            model on.
            input_encoder (optional): The input encoder to use. Defaults to None.
            output_encoder (optional): The output encoder to use. Defaults to None.
            savename (str, optional): The name to use when saving the results to a .mat
            savename (str, optional): The name to use when saving the results to a .mat
                                      Defaults to "".
        Returns:
            None
        """
        # TODO: TEST THIS FUNCTION
        self.model.eval()
        # create a tensor to store the prediction
        num_samples = len(data_loader.dataset)
        sample = next(iter(data_loader))
        data = sample["x"].to(self.device)
        if input_encoder is not None:
            data = input_encoder.decode(data)
        target = sample["y"].to(self.device)
        lentime = len(sample["meta_y"]["time"])
        size = [num_samples] + [d for d in target.shape if d != 1]
        in_size = [num_samples] + [d for d in data.shape if d != 1]

        # initialize
        pred = torch.zeros(size)
        truth = torch.zeros(size)
        in_data = torch.zeros(in_size)
        time_data = torch.zeros(num_samples, lentime)
        rmse = torch.zeros(num_samples)
        nrmse = torch.zeros(num_samples)
        rmse_masked = torch.zeros(num_samples)
        nrmse_masked = torch.zeros(num_samples)
        test_losses = {key: 0.0 for key in self.test_losses}
        with torch.no_grad():
            for b_idx, sample in enumerate(data_loader):
                data, target = sample["x"].to(self.device), sample["y"].to(self.device)
                output = self.model(data)

                # decode  if there is an output encoder
                if output_encoder is not None:
                    # decode the output
                    output = output_encoder.decode(output)

                # compute the loss
                # mask = data > 3 * data.std() + data.mean()
                mask = torch.stack([data[...,i] >= 3 * data[...,i].std() for i in range(data.shape[-1])],dim=-1)
                for key, loss_type in self.test_losses.items():
                    if key == "weight":
                        test_losses[key] += loss_type(output, target, mask).item()
                    else:
                        test_losses[key] += loss_type(output, target).item()

                # assign the values to the tensors
                pred[b_idx] = output.squeeze()
                in_data[b_idx] = data.squeeze()
                truth[b_idx] = target.squeeze()
                rmse[b_idx], nrmse[b_idx] = self.RMSE(output, target)
                rmse_masked[b_idx], nrmse_masked[b_idx] = self.RMSE(
                    output, target, mask=mask
                )
            # normalize test loss
            for key in test_losses:
                test_losses[key] /= len(data_loader.dataset)

            rmse = torch.mean(rmse)
            nrmse = torch.mean(nrmse)
            rmse_masked = torch.mean(rmse_masked)
            nrmse_masked = torch.mean(nrmse_masked)
            print(f"Length of Valid Data: {len(data_loader.dataset)}")
            # print the final loss
            num_trained = self.params.N * self.params.split[0]
            print(f"Final Loss for {int(num_trained)}: {test_losses['weight']} ")
            wandb.log(
                {
                    "Valid L2 Loss": test_losses["l2loss"],
                    "Valid H1 Loss": test_losses["h1loss"],
                    "Valid Wt Loss": test_losses["weight"],
                    "RMSE": rmse,
                    "NRMSE": nrmse,
                    "RMSE Masked": rmse_masked,
                    "NRMSE Masked": nrmse_masked,
                }
            )

        if len(savename) > 0:
            data_save = f"results/{self.params.path}/data/{savename}.mat"
            print(
                "Saving data as:",
                f"{os.getcwd()} {data_save}",
            )
            scipy.io.savemat(
                data_save,
                mdict={
                    "input": in_data.cpu().numpy(),
                    "pred": pred.cpu().numpy(),
                    "target": truth.cpu().numpy(),
                },
            )
        # flatten the time for plotting
        time_data = time_data.flatten()
        # if "Ints" in self.params.DATA_PATH:
        #     pred = pred.reshape(-1, *pred.shape[2:])
        #     in_data = in_data.reshape(-1, *in_data.shape[2:])
        #     truth = truth.reshape(-1, *truth.shape[2:])

        if self.params.doPlot:
            self.plot(
                prediction=pred,
                in_data=in_data,
                truth=truth,
                savename=f"{savename}_results",
                do_animate=do_animate,
            )
