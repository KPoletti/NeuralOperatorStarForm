import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO3d, FNO2d, FNO
from cliffordlayers.models.utils import partialclass
from cliffordlayers.models.models_2d import (
    CliffordNet2d,
    CliffordFourierBasicBlock2d,
)
from cliffordlayers.models.models_3d import (
    CliffordNet3d,
    CliffordFourierBasicBlock3d,
)
from neuralop.training.losses import LpLoss, H1Loss
from dataclasses import dataclass
import numpy as np
import pandas as pd
import input as param
import logging
import scipy
import time
import wandb
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
N = int((param.split[0] + param.split[1]) * param.N)
# create output folder
path = (
    f"SF_{param.NN}_{param.data_name}_ep{param.epochs}"
    f"_m{param.modes}_w{param.width}_S{param.S}"
    f"_E{param.encoder}"
    f"_N{N}"
)


# define the network architecture


def initializeNetwork(params: dataclass) -> nn.Module:
    """
    Initialize the model
    Input:
        params: input parameters from the input file
    Output:
        model: torch.nn.Module
    """
    models = {
        "FNO3d": FNO3d,
        "MNO": FNO2d,
        "FNO": FNO,
        "CNL2d": CliffordNet2d,
        "CNL3d": CliffordNet3d,
    }
    # TODO: allow MLP to be input parameter
    if params.NN == "FNO2d":
        model = FNO2d(
            n_modes_height=params.modes,
            n_modes_width=params.modes,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=True,
            mlp={"expansion": 0.5, "dropout": 0},
        )
    elif params.NN == "FNO3d":
        model = FNO3d(
            n_modes_depth=params.modes,
            n_modes_width=params.modes,
            n_modes_height=2,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=True,
            mlp={"expansion": 0.5, "dropout": 0},
        )
    elif params.NN == "MNO":
        model = FNO2d(
            n_modes_height=params.modes,
            n_modes_width=params.modes,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=True,
            mlp={"expansion": 0.5, "dropout": 0},
        )
    elif params.NN == "CNL2d":
        model = CliffordNet2d(
            g=[-1, -1],
            block=partialclass(
                "CliffordFourierBasicBlock2d",
                CliffordFourierBasicBlock2d,
                modes1=params.modes,
                modes2=params.modes,
            ),
            num_blocks=[1, 1, 1, 1],
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            hidden_channels=params.width,
            activation=F.gelu,
            norm=False,
            rotation=False,
        )
    elif params.NN == "CNL3d":
        model = CliffordNet3d(
            g=[1, 1, 1],
            block=partialclass(
                "CliffordFourierBasicBlock3d",
                CliffordFourierBasicBlock3d,
                modes1=params.modes,
                modes2=params.modes,
                modes3=params.modes,
            ),
            num_blocks=[1, 1, 1, 1],
            in_channels=4,
            out_channels=1,
            hidden_channels=params.width,
            activation=F.gelu,
            norm=False,
        )
    logger.debug(model)
    return model


def random_plot(
    input: torch.tensor, output: torch.tensor, target: torch.tensor, savename: str
):
    """Plots a time point of the input, output, and target"""
    # select random first and second dimension
    idx1 = np.random.randint(0, input.shape[0])
    idx2 = np.random.randint(0, input.shape[1])
    # check the shape of the input
    if len(input.shape) > 4:
        d = np.random.randint(0, input.shape[-1])
        input = input[..., d]
        output = output[..., d]
        target = target[..., d]

    # create 1 by 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # plot the input
    im0 = axs[0].imshow(input[idx1, idx2, :, :].cpu().detach().numpy())
    axs[0].set_title(f"Input: {idx1}, {idx2}")
    fig.colorbar(im0, ax=axs[0])
    # plot the output
    im1 = axs[1].imshow(output[idx1, idx2, :, :].cpu().detach().numpy())
    axs[1].set_title(f"Output: {idx1}, {idx2}")
    fig.colorbar(im1, ax=axs[1])
    # plot the target
    im2 = axs[2].imshow(target[idx1, idx2, :, :].cpu().detach().numpy())
    axs[2].set_title(f"Target: {idx1}, {idx2}")
    fig.colorbar(im2, ax=axs[2])
    # save the figure
    plt.savefig(f"results/{path}/plots/random_plot_{savename}_id1{idx1}_id2{idx2}.png")


class Trainer(object):
    def __init__(
        self, model: nn.Module, params: dataclass, device: torch.device
    ) -> None:
        self.model = model
        self.params = params
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=params.scheduler_step,
            gamma=params.scheduler_gamma,
        )
        self.loss = params.loss_fn

    def dissipativeRegularizer(
        self, x: torch.Tensor, device: torch.device
    ) -> torch.nn.MSELoss:
        # TODO: Test this function

        # DISSIPATIVE REGULARIZATION
        x_diss = torch.tensor(
            self.params.sampling_fn(x.shape[0], self.params.radii, x.shape[1:]),
            dtype=torch.float,
        ).to(device)
        assert x_diss.shape == x.shape
        y_diss = self.params.target_fn(x_diss, self.params.scale_down).to(device)
        out_diss = self.model(x_diss).reshape(-1, self.params.out_dim)
        diss_loss = (
            (1 / (self.params.S**2))
            * self.params.loss_weight
            * self.params.dissloss(out_diss, y_diss.reshape(-1, self.params.out_dim))
        )  # weighted by 1 / (S**2)
        return diss_loss

    def batchLoop(self, train_loader, output_encoder=None, epoch: int = 1):
        # initialize loss
        train_loss = 0
        # loss = 0
        diss_loss = 0
        diss_loss_l2 = 0
        # select a random batch
        rand_point = np.random.randint(0, len(train_loader))
        torch.autograd.set_detect_anomaly(True)
        for batch_idx, sample in enumerate(train_loader):
            data, target = sample["x"].to(self.device), sample["y"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(data)
            if epoch == 0 and batch_idx == 0:
                logger.debug(f"Batch Data Shape: {data.shape}")
                logger.debug(f"Batch Target Shape: {target.shape}")
                logger.debug(f"Output Data Shape: {output.shape}")
            if batch_idx == rand_point and epoch == 33:
                random_plot(data, output, target, savename="train")

            # decode  if there is an output encoder
            if output_encoder is not None:
                # decode the target
                target = output_encoder.decode(target)
                # decode the output
                output = output_encoder.decode(output)

            if batch_idx == rand_point and epoch == 33:
                random_plot(data, output, target, savename="train_decoded")
            # compute the loss
            loss = self.loss(output.float(), target)
            if len(loss.shape) > 1:
                logger.debug(f"Loss Shape: {loss.shape}")
                logger.debug(f"Loss Value: {loss.item()}")

            # add regularizer for MNO
            if self.params.NN == "MNO":
                diss_loss = self.dissipativeRegularizer(data, self.device)
                diss_loss_l2 += diss_loss.item()
                loss += diss_loss
            # Backpropagate the loss
            loss.backward()

            self.optimizer.step()
            train_loss += loss.item()
        return train_loss, loss

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
        # initialize the loss function
        self.loss = self.params.loss_fn
        # define a test loss function that will be the same regardless of training
        if sweep:
            test_loss_fn = LpLoss(d=self.params.d, p=2, reduce_dims=(0, 1))
        else:
            test_loss_fn = self.loss
        # start training
        for epoch in range(self.params.epochs):
            epoch_timer = time.time()

            # set to train mode
            self.model.train()
            train_loss, loss = self.batchLoop(
                train_loader, output_encoder=output_encoder, epoch=epoch
            )

            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for sample in test_loader:
                    data, target = sample["x"].to(self.device), sample["y"].to(
                        self.device
                    )
                    output = self.model(data)
                    # decode  if there is an output encoder
                    if output_encoder is not None:
                        output = output_encoder.decode(output)
                        # target = output_encoder.decode(target)
                        # do not decode the test target because the test data is not encoded
                    test_loss += test_loss_fn(output, target).item()

            test_loss /= len(test_loader.dataset)
            train_loss /= len(train_loader.dataset)
            self.scheduler.step()

            epoch_timer = time.time() - epoch_timer
            logger.info(
                f"Epoch: {epoch+1}/{self.params.epochs} \t"
                f"Took: {epoch_timer:.2f} \t"
                f"Training Loss: {train_loss:.6f} \t"
                f"Test Loss: {test_loss:.6f}"
            )
            print(
                f"Epoch: {epoch+1}/{self.params.epochs} \t"
                f"Took: {epoch_timer:.2f} \t"
                f"Training Loss: {train_loss:.6f} \t"
                f"Test Loss: {test_loss:.6f}"
            )
            wandb.log({"Test-Loss": test_loss, "Train-Loss": train_loss})
        try:
            if self.params.saveNeuralNetwork:
                torch.save(self.model, f"results/{path}/models/{self.params.NN}")
        except:
            logger.error("Failed to save model")

    def densityRMSE(
        self,
        prediction,
        truth,
    ):
        """
        Calculate the RMSE as a function of density
        Input:
            prediction: torch.tensor
            input: torch.tensor
            truth: torch.tensor
        Output:
            bins: np.array
            mean_RMSE: np.array
            std_RMSE: np.array
        """
        # change tensor to numpy array
        prediction = prediction.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()
        # create bins of the data
        numBins = 100
        # create number of time steps to analyze
        numSteps = 20
        # create array to store the RMSE
        RMSE_array = np.zeros((numSteps, numBins))
        bin_array = np.zeros((numSteps, numBins))

        # create a list of random time steps
        randInd = np.random.randint(0, prediction.shape[0], numSteps)

        for j, ind in enumerate(randInd):
            # create bins of the data
            _, bins = np.histogram(truth[ind, :, :, 0], bins=numBins)
            # create a mask for data
            mask = np.digitize(truth[ind, :, :, 0], bins)
            bin_array[j, :] = bins[1:]
            # loop over the bins
            for i in range(1, numBins + 1):
                truthBin = truth[ind, :, :, 0][mask == i]
                predictionBin = prediction[ind, :, :, 0][mask == i]
                # compute RMSE
                RMSE_array[j, i - 1] = np.sqrt(np.mean((predictionBin - truthBin) ** 2))
        # compute the mean and std of the RMSE
        mean_RMSE = np.mean(RMSE_array, axis=0)
        std_RMSE = np.std(RMSE_array, axis=0)
        return bins, mean_RMSE, std_RMSE

    def plot(self, timeData, prediction, input, truth, savename=""):
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

        import matplotlib.pyplot as plt
        import seaborn as sns

        # set the style
        sns.set_style("darkgrid")

        ############ RMSE ############
        num_Dims = len(prediction.shape)
        ind = -1
        if num_Dims == 5:
            # select a random int between 0 and
            ind = np.random.randint(0, 2)
            t_ind = np.random.randint(0, self.params.dN - 1)
            # reduce the data to only that index
            # prediction = prediction[:, :, :, ind, :].permute(0, 3, 1, 2).flatten(0, 1)
            prediction = prediction[:, :, :, ind, t_ind]
            truth = truth[:, :, :, ind, t_ind]
            input = input[:, :, :, ind, t_ind]
            # and (n,x,y,t) flatten to (t*n, x, y)
            num_Dims = len(prediction.shape)

        dims_to_rmse = tuple(range(-num_Dims + 1, 0))

        # compute RMSE
        rmse = torch.sqrt(torch.mean((prediction - truth) ** 2, dim=dims_to_rmse))
        # normalize RMSE
        rmse /= torch.sqrt(torch.mean(truth**2, dim=dims_to_rmse))
        # compute RMSE for each time step
        relativeRMSE = torch.sqrt(torch.mean((input - truth) ** 2, dim=dims_to_rmse))
        # normalize RMSE
        relativeRMSE /= torch.sqrt(torch.mean(truth**2, dim=dims_to_rmse))

        try:
            time = timeData["t+0"]["time"].values
        except NameError:
            time = np.arange(0, prediction.shape[0])
            logger.info("Failed time data")
        except KeyError:
            time = timeData["t"]["time"].values

        # plot RMSE for each time step
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(time, rmse, ".", label="Neural RMSE")
        ax.plot(time, relativeRMSE, ".", label="RMSE between time steps")
        ax.set_xlabel("Time (Myr)")
        ax.set_ylabel("Normalized  RMSE")
        ax.set_title(f"RMSE Over Time for index {ind}, t_0 {t_ind}")
        ax.legend()
        im = wandb.Image(fig)
        wandb.log({"RMSE": im})
        if len(savename) > 0:
            fig.savefig(f"results/{path}/plots/{savename}_RMSE.png")
        plt.close(fig)

        ############ RMSE Vs Density ############
        # # compute the RMSE as a function of density
        # predBins, predMean, predSTD = self.densityRMSE(prediction, truth)
        # inputBins, inputMean, inputSTD = self.densityRMSE(input, truth)
        # # plot the RMSE as a function of density
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.loglog(predBins[:-1], predMean, label="Neural RMSE")
        # ax.fill_between(
        #     predBins[:-1], predMean - predSTD, predMean + predSTD, alpha=0.2
        # )
        # ax.loglog(inputBins[:-1], inputMean, label="RMSE between time steps")
        # ax.fill_between(
        #     inputBins[:-1], inputMean - inputSTD, inputMean + inputSTD, alpha=0.2
        # )
        # ax.set_xlabel(r"Column Density $  \left[\frac{g}{cm^2} \right]$")
        # ax.set_ylabel("RMSE")
        # ax.legend()
        # ax.set_title("RMSE as a function of density")
        # im = wandb.Image(fig)
        # wandb.log({"RMSE vs Density": im})
        # if len(savename) > 0:
        #     fig.savefig(f"results/{path}/plots/{savename}_RMSE_vs_density.png")
        # plt.close(fig)

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        timeData: pd.DataFrame,
        input_encoder=None,
        output_encoder=None,
        savename: str = "",
    ) -> None:
        """
        Evaluate the model
        Input:
            data_loader: torch.utils.data.DataLoader
        Output:
            None
        """
        # TODO: TEST THIS FUNCTION
        self.model.eval()
        # create a tensor to store the prediction
        num_samples = len(data_loader.dataset)
        for batchidx, sample in enumerate(data_loader):
            data, target = sample["x"].to(self.device), sample["y"].to(self.device)
            break

        size = [num_samples] + [d for d in data.shape if d != 1]
        pred = torch.zeros(size)
        rePred = torch.zeros(size)
        input = torch.zeros(size)
        truth = torch.zeros(size)
        test_loss = 0
        re_loss = 0
        rand_point = np.random.randint(0, len(data_loader))

        with torch.no_grad():
            for batchidx, sample in enumerate(data_loader):
                data, target = sample["x"].to(self.device), sample["y"].to(self.device)
                # apply the input encoder
                output = self.model(data)
                # apply the model to previous output
                if batchidx == 0:
                    reData = output.clone()
                elif batchidx > 0:
                    if input_encoder is not None:
                        reData = input_encoder.encode(reData)
                    # apply the model to previous output
                    reData = self.model(reData)

                if batchidx == rand_point:
                    random_plot(data, output, target, "valid")
                    random_plot(data, reData, target, "rolling")

                # decode  if there is an output encoder
                if output_encoder is not None:
                    # decode the output
                    output = output_encoder.decode(output)
                    # decode the multiple applications of the model
                    reData = output_encoder.decode(reData)
                # compute the loss
                re_loss += self.loss(reData, target).item()
                test_loss += self.loss(output, target).item()

                if input_encoder is not None:
                    data = input_encoder.decode(data)

                if batchidx == rand_point + 1:
                    random_plot(data, output, target, savename="valid_decoded")
                    random_plot(data, reData, target, "rolling_decoded")

                # assign the values to the tensors
                pred[batchidx] = output
                input[batchidx] = data
                truth[batchidx] = target
                rePred[batchidx] = reData
            # normalize test loss
            test_loss /= len(data_loader.dataset)
            re_loss /= len(data_loader.dataset)
            print(f"Length of Valid Data: {len(data_loader.dataset)}")
            # print the final loss
            numTrained = self.params.N * self.params.split[0]
            print(f"Final Loss for {int(numTrained)}: {test_loss} ")
            wandb.log({"Valid Loss": test_loss, "Reapply Loss": re_loss})
        print(f"Saving data as {os.getcwd()} results/{path}/data/{savename}.mat")
        if len(savename) > 0:
            print(f"Saving data as {os.getcwd()} results/{path}/data/{savename}.mat")
            scipy.io.savemat(
                f"results/{path}/data/{savename}.mat",
                mdict={
                    "input": input.cpu().numpy(),
                    "pred": pred.cpu().numpy(),
                    "target": truth.cpu().numpy(),
                    "rePred": rePred.cpu().numpy(),
                },
            )
        if self.params.doPlot:
            self.plot(
                timeData=timeData[:num_samples],
                prediction=pred,
                input=input,
                truth=truth,
                savename=savename,
            )
