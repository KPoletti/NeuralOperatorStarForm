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
from torch import nn

from neuralop import LpLoss, H1Loss
from neuralop.datasets.transforms import PositionalEmbedding2D


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
    target: torch.Tensor, output: torch.Tensor, meta: dict, save: str, idx, device
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
    if "FNO3d" in save or "CNL2d" in save or "RNN" in save:
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


class Trainer(object):
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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay,
            amsgrad=True,
        )
        if self.device == 0:
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            print(f"Adam LR {lr}")
        self.loss = params.loss_fn
        self.plot_path = f"results/{self.params.path}/plots"
        self.save_every = save_every
        self.epochs_run = 0
        self.ckp_path = ckp_path
        self.do_animate = (self.device != 0) * self.params.use_ddp
        print(self.do_animate)
        # initialize the loss function
        ntrain = int((self.params.split[0] * self.params.N))
        self.loss = self.params.loss_fn
        iterations = (
            self.params.epochs
            * (ntrain // self.params.batch_size)
            // self.params.cosine_step
        )
        if self.params.use_ddp:
            self.num_gpus = torch.cuda.device_count()

        if os.path.exists(self.ckp_path) and params.use_ddp:
            self._load_snapshot(self.ckp_path)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=iterations,
            eta_min=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=iterations,
            eta_min=1e-8,
        )
        # define a test loss function that will be the same regardless of training
        self.test_loss_lp = LpLoss(d=self.params.d, p=2, reduce_dims=(0, 1))
        self.test_loss_h1 = H1Loss(d=self.params.d, reduce_dims=(0, 1))
        if self.params.NN == "RNN":
            self.full_loss = H1Loss(d=self.params.d + 1, reduce_dims=(0, 1))

    def dissipativeRegularizer(
        self, data: torch.Tensor, device: torch.device
    ) -> torch.nn.MSELoss:
        """
        Computes the dissipative regularization loss for a given input tensor x.

        Args:
            x (torch.Tensor): The input tensor to compute the dissipative regularization
            loss for.
            device (torch.device): The device to use for computation.

        Returns:
            torch.nn.MSELoss: The dissipative regularization loss.
        """
        # TODO: Test this function

        # DISSIPATIVE REGULARIZATION
        x_diss = torch.tensor(
            self.params.sampling_fn(data.shape[0], self.params.radii, data.shape[1:]),
            dtype=torch.float,
        ).to(self.device)
        assert x_diss.shape == data.shape
        y_diss = self.params.target_fn(x_diss, self.params.scale_down).to(self.device)
        out_diss = self.model(x_diss).reshape(-1, self.params.out_dim)
        diss_loss = (
            (1 / (self.params.S**2))
            * self.params.loss_weight
            * self.params.dissloss(out_diss, y_diss.reshape(-1, self.params.out_dim))
        )  # weighted by 1 / (S**2)
        return diss_loss

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
        idx = 0
        loss = 0
        diss_loss = 0
        train_loss = 0
        diss_loss_l2 = 0
        # select a random batch
        rand_point = np.random.randint(0, len(train_loader))
        torch.autograd.set_detect_anomaly(True)
        counter = 0
        for batch_idx, sample in enumerate(train_loader):
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

            if batch_idx == rand_point and epoch == self.save_every:
                savename = f"{self.plot_path}/Train_decoded_b{batch_idx}_ep{epoch}"
                dataVisibleCheck(
                    target,
                    output,
                    sample["meta_y"],
                    f"{savename}",
                    idx,
                    device=self.do_animate,
                )
            # compute the loss
            loss = self.loss(output.float(), target)
            # add regularizer for MNO
            if self.params.NN == "MNO":
                diss_loss = self.dissipativeRegularizer(data, self.device)
                diss_loss_l2 += diss_loss.item()
                loss += diss_loss
            # Backpropagate the loss
            loss.backward()
            self.optimizer.step()
            if hasattr(self, "scheduler"):
                self.scheduler.step()
            train_loss += loss.item()
            self.optimizer.zero_grad(set_to_none=True)
            if (
                loss.item() - train_loss
            ) ** 2 >= 20 * train_loss**2 and batch_idx > 0:
                savename = f"{self.plot_path}/Train_decoded_Jump_b{batch_idx}_ep{epoch}"
                dataVisibleCheck(
                    target,
                    output,
                    sample["meta_y"],
                    f"{savename}target",
                    idx,
                    device=self.do_animate,
                )
                counter += 1
            if counter >= 0.5 * self.params.batch_size:
                raise ValueError("Training loss is growing too much.")

        return train_loss, loss

    def recurrent_loop(self, train_loader, output_encoder=None, epoch: int = 1):
        step = self.params.T_in
        # initialize loss
        train_loss = 0
        full_loss = 0
        # select a random batch
        rand_point = np.random.randint(0, len(train_loader))
        torch.autograd.set_detect_anomaly(True)
        for batch_idx, sample in enumerate(train_loader):
            loss = 0
            data = sample["x"].to(self.device).squeeze(-1)
            target = sample["y"].to(self.device)
            for t in range(0, target.shape[-1], step):
                targ_t = target[..., t : t + step].squeeze(-1)
                pred_t = self.model(data)

                loss += self.loss(pred_t.float(), targ_t)
                if t == 0:
                    pred = pred_t.unsqueeze(-1)
                else:
                    pred = torch.cat((pred, pred_t.unsqueeze(-1)), dim=-1)
                data = pred_t

            train_loss += loss.item()
            full_loss += self.full_loss(pred, target).item()
            # Backpropagate the loss
            self.optimizer.zero_grad(set_to_none=True)

            loss.backward()
            self.optimizer.step()

            if hasattr(self, "scheduler"):
                self.scheduler.step()
            if batch_idx == rand_point and epoch == self.save_every:
                savename = f"{self.plot_path}/Train_decoded_b{batch_idx}_ep{epoch}"
                dataVisibleCheck(
                    target,
                    pred,
                    sample["meta_y"],
                    f"{savename}",
                    0,
                    device=self.do_animate,
                )
        return train_loss, loss

    def recurrent_eval(self, test_loader, output_encoder=None, epoch: int = 1):
        step = self.params.T_in
        # initialize loss
        test_h1 = 0
        test_l2 = 0
        full_test = 0
        # select a random batch
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):
                h1_loss = 0
                l2_loss = 0
                data = sample["x"].to(self.device).squeeze(-1)
                target = sample["y"].to(self.device)
                for t in range(0, target.shape[-1], step):
                    targ_t = target[..., t : t + step].squeeze(-1)
                    pred_t = self.model(data)

                    h1_loss += self.test_loss_h1(pred_t.float(), targ_t)
                    l2_loss += self.test_loss_lp(pred_t.float(), targ_t)

                    if t == 0:
                        pred = pred_t.unsqueeze(-1)
                    else:
                        pred = torch.cat((pred, pred_t.unsqueeze(-1)), dim=-1)
                    data = pred_t

                test_h1 += h1_loss.item()
                test_l2 += l2_loss.item()

                full_test += self.full_loss(pred, target).item()

        return test_h1, test_l2, full_test

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
        loss_old = 500

        # start training
        for epoch in range(self.epochs_run, self.params.epochs):
            epoch_timer = time.time()

            # set to train mode
            self.model.train()
            if "RNN" in self.params.NN:
                train_loss, _ = self.recurrent_loop(train_loader, output_encoder, epoch)
            else:
                train_loss, _ = self.batchLoop(train_loader, output_encoder, epoch)

            if (
                self.params.use_ddp
                and self.device == 0
                and epoch % self.save_every == 0
            ):
                self._save_snapshot(epoch)
            train_loss /= train_data_length
            loss_ratio = (train_loss - loss_old) / loss_old
            loss_old = train_loss
            self.model.eval()
            test_lp = 0
            test_h1 = 0
            if self.params.NN == "RNN":
                test_h1, test_lp, full_test = self.recurrent_eval(
                    test_loader, output_encoder, epoch
                )

            else:
                rand_point = np.random.randint(0, test_data_length)

                for batch_idx, sample in enumerate(test_loader):
                    data = sample["x"].to(self.device)
                    target = sample["y"].to(self.device)

                    output = self.model(data)

                    # decode  if there is an output encoder
                    if output_encoder is not None:
                        output = output_encoder.decode(output)
                        # do not decode the test target because test data isn't encode
                    if (batch_idx == rand_point and epoch == self.save_every) or (
                        batch_idx == rand_point and loss_ratio >= 3
                    ):
                        idx = 0
                        if data.shape[0] > 1:
                            idx = np.random.randint(0, data.shape[0] - 1)
                        savename = f"{self.plot_path}/Test_decoded_R{loss_ratio:1.2f}_ep{epoch}"
                        dataVisibleCheck(
                            target,
                            output,
                            sample["meta_y"],
                            f"{savename}",
                            idx,
                            device=self.do_animate,
                        )
                    test_lp += self.test_loss_lp(output, target).item()
                    test_h1 += self.test_loss_h1(output, target).item()

            test_lp /= test_data_length
            test_h1 /= test_data_length

            # self.scheduler1.step()

            epoch_timer = time.time() - epoch_timer
            logger.info(
                "Epoch: %d/%d \t Took: %.2f \t Training Loss: %.6f \t Test L2 Loss: %.6f \t Test H1 Loss: %.6f",
                epoch + 1,
                self.params.epochs,
                epoch_timer,
                train_loss,
                test_lp,
                test_h1,
            )
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            if self.device == 0:
                print(f"Adam LR {lr}")

            print(
                f"Epoch: {epoch+1}/{self.params.epochs} \t"
                f"Took: {epoch_timer:.2f} \t"
                f"Training Loss: {train_loss:.6f} \t"
                f"Test L2 Loss: {test_lp:.6f}\t"
                f"Test H1 Loss: {test_h1:.6f}\t"
            )
            wandb.log(
                {
                    "Test-Loss-L2": test_lp,
                    "Test-Loss-H1": test_h1,
                    "Train-Loss": train_loss,
                    "Learn-Rate": lr,
                }
            )

        if self.params.saveNeuralNetwork:
            modelname = f"results/{self.params.path}/models/{self.params.NN}.pt"
            if self.params.use_ddp:
                torch.save(self.model.module.state_dict(), modelname)
            else:
                torch.save(self.model.state_dict(), modelname)

    def rmse_plot(
        self,
        ind,
        num_dims,
        time_data,
        truth,
        prediction,
        in_data,
        savename,
    ):
        num_samples = prediction.shape[0]
        dims_to_rmse = tuple(range(-num_dims + 1, 0))

        # compute RMSE
        rmse = torch.sqrt(torch.mean((prediction - truth) ** 2, dim=dims_to_rmse))
        rmse_mean = torch.mean(rmse)
        # normalize RMSE
        rmse /= torch.sqrt(torch.mean(truth**2, dim=dims_to_rmse))
        nrmse_mean = torch.mean(rmse)
        # wandb.log({"RMSE": rmse_mean, "Relative RMSE": nrmse_mean})

        # compute RMSE for each time step
        relative_rmse = torch.sqrt(torch.mean((in_data - truth) ** 2, dim=dims_to_rmse))
        relative_rmse /= torch.sqrt(torch.mean(truth**2, dim=dims_to_rmse))

        time_data = time_data.detach().numpy()
        rmse = rmse.detach().numpy()
        relative_rmse = relative_rmse.detach().numpy()

        # sort rmse based on time to take mean
        ind_sort = time_data.argsort()
        sorted_rmse = rmse[ind_sort[::-1]]
        sorted_time = time_data[ind_sort[::-1]]
        sorted_relt = relative_rmse[ind_sort[::-1]]

        def moving_average(x, w):
            return np.convolve(x, np.ones(w), "valid") / w

        avg_rmse = moving_average(sorted_rmse, 50)
        avg_relt = moving_average(sorted_relt, 50)

        # set the style
        sns.set_style("darkgrid")
        # plot RMSE for each time step
        fig, axis = plt.subplots(1, 1, figsize=(10, 10))
        axis.plot(
            time_data[: 3 * num_samples],
            rmse[: 3 * num_samples],
            "b.",
            alpha=0.2,
            label="Neural RMSE",
        )
        axis.plot(
            sorted_time[: avg_rmse.shape[0]],
            avg_rmse,
            color="b",
            label=" Average Neural RMSE",
        )
        axis.plot(
            time_data[: 3 * num_samples],
            relative_rmse[: 3 * num_samples],
            "g.",
            alpha=0.2,
            label="RMSE between time steps",
        )
        axis.plot(
            sorted_time[: avg_rmse.shape[0]],
            avg_relt,
            "g",
            label="Average RMSE between time steps",
        )
        axis.set_xlabel("Time (Myr)")
        axis.set_ylabel("Normalized  RMSE")
        axis.set_title(f"RMSE Over Time for index {ind}, t_0 {time_data[0]}")
        axis.legend()
        img = wandb.Image(fig)
        wandb.log({"RMSE": img})
        if len(savename) > 0:
            fig.savefig(f"{self.plot_path}/{savename}_RMSE.png")
        plt.close(fig)
        plt.close()

    def plot(
        self,
        time_data,
        prediction,
        in_data,
        truth,
        roll,
        mass,
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
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        ############ RMSE ############
        num_dims = len(prediction.shape)
        num_samples = prediction.shape[0]
        ind = -1
        ind_grid = -1
        ind_tim = -1
        ind_dim = -1

        if num_dims == 5:
            # find which dimension is dN//2
            for i in range(1, len(prediction.shape)):
                if prediction.shape[i] == self.params.output_channels:
                    ind_dim = i
                if prediction.shape[i] == self.params.S:
                    ind_grid = i
                if prediction.shape[i] == self.params.dN // 2:
                    ind_tim = i
            ind_grid -= 1

            # select a random int between 0 and
            ind = 0
            # reduce the data to only that indices for ind_dim and ind_tim
            prediction = prediction.index_select(ind_dim, torch.tensor(ind))
            prediction = prediction.permute(0, ind_tim, ind_grid, ind_grid + 1, ind_dim)

            truth = truth.index_select(ind_dim, torch.tensor(ind))
            truth = truth.permute(0, ind_tim, ind_grid, ind_grid + 1, ind_dim)

            roll = roll.index_select(ind_dim, torch.tensor(ind))
            roll = roll.permute(0, ind_tim, ind_grid, ind_grid + 1, ind_dim)

            in_data = in_data.index_select(ind_dim, torch.tensor(ind))
            in_data = in_data.permute(0, ind_tim, ind_grid, ind_grid + 1, ind_dim)

        # and (n,x,y,t) flatten to (t*n, x, y)
        prediction = prediction.squeeze().flatten(0, 1)
        truth = truth.squeeze().flatten(0, 1)
        roll = roll.squeeze().flatten(0, 1)
        in_data = in_data.squeeze().flatten(0, 1)

        num_dims = len(prediction.shape)
        if do_animate:
            Animation_true_pred_error(
                truthData=truth,
                predcData=prediction,
                savePath=f"{self.plot_path}/out_",
                out_times=time_data,
                mass="",
            )
            Animation_true_pred_error(
                truthData=truth,
                predcData=roll,
                savePath=f"{self.plot_path}/roll_",
                out_times=time_data,
                mass="",
            )
        self.log_RMSE(prediction, truth)
        self.rmse_plot(
            ind,
            num_dims,
            time_data,
            truth,
            roll,
            in_data,
            f"{savename}_rolling",
        )
        self.rmse_plot(
            ind,
            num_dims,
            time_data,
            truth,
            prediction,
            in_data,
            f"{savename}",
        )

    def log_RMSE(self, prediction, truth):
        num_dims = len(prediction.shape)
        dims_to_rmse = tuple(range(-num_dims + 1, 0))

        # compute RMSE
        rmse = torch.sqrt(torch.mean((prediction - truth) ** 2, dim=dims_to_rmse))
        rmse_mean = torch.mean(rmse)
        # normalize RMSE
        rmse /= torch.sqrt(torch.mean(truth**2, dim=dims_to_rmse))
        nrmse_mean = torch.mean(rmse)
        wandb.log({"RMSE": rmse_mean, "Relative RMSE": nrmse_mean})

    def evaluate_RNN(
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
        data_length = len(data_loader.dataset)
        if "RNN" in self.params.NN:
            data_length *= self.params.T
        # create a tensor to store the prediction
        num_samples = len(data_loader.dataset)
        sample = next(iter(data_loader))
        data = sample["x"].to(self.device)
        target = sample["y"].to(self.device)
        lentime = len(sample["meta_y"]["time"]) - 1
        size = [num_samples] + [d for d in target.shape if d != 1]

        # initialize
        pred = torch.zeros(size)
        roll = torch.zeros(size)
        truth = torch.zeros(size)
        in_data = torch.zeros(size)

        # in_data = torch.zeros([*size[:-1], self.params.T_in])

        time_data = torch.zeros(num_samples, lentime)

        test_pred = 0
        test_roll = 0
        full_test = 0
        roll_t = 0
        step = self.params.T_in

        mass = "null"
        mass_list = []
        rand_point = np.random.randint(0, len(data_loader))
        transform = None
        if self.params.positional_encoding:
            transform = PositionalEmbedding2D(self.params.grid_boundaries, 0)
        with torch.no_grad():
            for batchidx, sample in enumerate(data_loader):
                loss_pred = 0
                loss_roll = 0

                meta_data = sample["meta_x"]
                cur_mass = meta_data["mass"][0]
                target = sample["y"].to(self.device)
                data = sample["x"].to(self.device).squeeze(-1)
                time_data[batchidx] = torch.tensor(sample["meta_y"]["time"][1:])
                if cur_mass != mass:
                    roll_t = data.clone()
                    logging.info(f"Updating mass from M{mass} to M{cur_mass}")
                    print(f"Updating mass from M{mass} to M{cur_mass}")
                    mass = cur_mass
                    roll_steps = 1
                else:
                    roll_steps += 1
                for t in range(0, target.shape[-1], step):
                    targ_t = target[..., t : t + step].squeeze(-1)
                    pred_t = self.model(data)
                    # apply the model to previous output
                    roll_t = self.model(roll_t)

                    if t == 0:
                        pred_full = pred_t.unsqueeze(-1)
                        roll_full = roll_t.unsqueeze(-1)
                    else:
                        pred_full = torch.cat((pred_full, pred_t.unsqueeze(-1)), dim=-1)
                        roll_full = torch.cat((roll_full, roll_t.unsqueeze(-1)), dim=-1)

                    loss_pred += self.loss(pred_t.float(), targ_t)
                    loss_roll += self.loss(roll_t.float(), targ_t)

                    data = pred_t

                if roll_steps == 5:
                    mass = "Null"
                    roll_steps = 1
                test_pred += loss_pred.item()
                test_roll += loss_roll.item()
                # wandb.log(
                #     {
                #         "Valid-Batch Loss": loss_pred.item(),
                #         "Roll-Batch Loss": loss_roll.item(),
                #     }
                # )
                full_test += self.full_loss(pred_full, target).item()
                # assign the values to the tensors
                pred[batchidx] = pred_full
                in_data[batchidx] = torch.cat(
                    (sample["x"].to(self.device), pred_full[..., :-1]), dim=-1
                )
                truth[batchidx] = target
                roll[batchidx] = roll_full

            # normalize test loss
            test_pred /= data_length
            test_roll /= data_length
            print(f"Length of Valid Data: {len(data_loader.dataset)}")
            # print the final loss
            num_trained = self.params.N * self.params.split[0]
            print(f"Final Loss for {int(num_trained)}: {test_pred} ")
            print(f"Reapply Loss: {test_roll}")
            wandb.log({"Valid Loss": test_pred, "Reapply Loss": test_roll})

        time_data = time_data.flatten()
        print(f"pred shape: {pred.shape}")
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
                    "rePred": roll.cpu().numpy(),
                },
            )

        if "RNN" in self.params.NN and len(pred.shape) == 4:
            pred = pred.unsqueeze(1)
            in_data = in_data.unsqueeze(1)
            truth = truth.unsqueeze(1)
            roll = roll.unsqueeze(1)
        if self.params.doPlot:
            self.plot(
                time_data=time_data,
                prediction=pred,
                in_data=in_data,
                truth=truth,
                roll=roll,
                mass=mass_list,
                savename=f"{savename}_results",
                do_animate=do_animate,
            )

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
        target = sample["y"].to(self.device)
        lentime = len(sample["meta_x"]["time"])
        size = [num_samples] + [d for d in target.shape if d != 1]

        # initialize
        pred = torch.zeros(size)
        roll = torch.zeros(size)
        truth = torch.zeros(size)
        in_data = torch.zeros(size)
        time_data = torch.zeros(num_samples, lentime)

        re_loss = 0
        test_loss = 0
        roll_batch = 0

        mass = "null"
        mass_list = []
        rand_point = np.random.randint(0, len(data_loader))
        transform = None
        if self.params.positional_encoding:
            transform = PositionalEmbedding2D(self.params.grid_boundaries, 0)
        with torch.no_grad():
            for batchidx, sample in enumerate(data_loader):
                data, target = sample["x"].to(self.device), sample["y"].to(self.device)
                meta_data = sample["meta_x"]
                cur_mass = meta_data["mass"][0]
                time_data[batchidx] = torch.tensor(sample["meta_y"]["time"][1:])
                output = self.model(data)
                # apply the model to previous output
                if batchidx == 0 or cur_mass != mass:
                    roll_batch = output.clone()
                    logging.info(f"Updating mass from M{mass} to M{cur_mass}")
                    print(f"Updating mass from M{mass} to M{cur_mass}")
                    mass = cur_mass
                    roll_steps = 1
                elif batchidx > 0:
                    # apply the model to previous output
                    roll_steps += 1
                    roll_batch = self.model(roll_batch)

                # decode  if there is an output encoder
                if output_encoder is not None:
                    # decode the output
                    output = output_encoder.decode(output)
                    # decode the multiple applications of the model
                    roll_out = output_encoder.decode(roll_batch)
                if batchidx == rand_point:
                    save_plot = f"{self.plot_path}/Valid_"
                    idx = 0
                    dataVisibleCheck(
                        target,
                        output,
                        sample["meta_y"],
                        f"{save_plot}output",
                        idx,
                        device=self.do_animate,
                    )
                    dataVisibleCheck(
                        target,
                        roll_out,
                        sample["meta_y"],
                        f"{save_plot}rolling",
                        idx,
                        device=self.do_animate,
                    )

                if roll_steps == 5:
                    mass = "Null"
                    roll_steps = 1
                # compute the loss
                re_loss += self.test_loss_lp(roll_out, target).item()
                test_loss += self.test_loss_lp(output, target).item()
                if transform is not None:
                    data = data[:, 0:5, ...]
                if input_encoder is not None:
                    data = input_encoder.decode(data)

                # assign the values to the tensors
                pred[batchidx] = output
                in_data[batchidx] = data
                truth[batchidx] = target
                roll[batchidx] = roll_out

                # if input_encoder is not None:
                #     roll_batch = input_encoder.encode(roll_out)
                if transform is not None:
                    roll_batch = transform(roll_batch[0, ...].cpu()).to(self.device)
                    roll_batch = roll_batch.unsqueeze(0)

                mass_list = mass_list + [cur_mass] * lentime

            # normalize test loss
            test_loss /= len(data_loader.dataset)
            re_loss /= len(data_loader.dataset)
            print(f"Length of Valid Data: {len(data_loader.dataset)}")
            # print the final loss
            num_trained = self.params.N * self.params.split[0]
            print(f"Final Loss for {int(num_trained)}: {test_loss} ")
            print(f"Reapply Loss: {re_loss}")

            wandb.log({"Valid Loss": test_loss, "Reapply Loss": re_loss})

        # flatten the time for plotting
        time_data = time_data.flatten()
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
                    "rePred": roll.cpu().numpy(),
                },
            )
        if self.params.doPlot:
            self.plot(
                time_data=time_data,
                prediction=pred,
                in_data=in_data,
                truth=truth,
                roll=roll,
                mass=mass_list,
                savename=f"{savename}_results",
                do_animate=do_animate,
            )
