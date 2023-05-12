import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO3d, FNO2d, FNO
from neuralop.training.losses import LpLoss, H1Loss
from dataclasses import dataclass
import numpy as np
import pandas as pd
import input as param
import logging
import scipy
import time
import wandb

logger = logging.getLogger(__name__)

# grab the ending of the density file with .pt
densityProfile = param.TRAIN_PATH.split("density")[-1].split(".")[0]
N = int((param.split[0] + param.split[1]) * param.N)
# create output folder
path = (
    f"SF_{param.NN}_{param.data_name}_ep{param.epochs}"
    f"_m{param.modes}_w{param.width}_S{param.S}"
    f"_E{param.encoder}"
    f"_N{N}"
)

if not os.path.exists(f"results/{path}/logs"):
    os.makedirs(f"results/{path}/logs")
    os.makedirs(f"results/{path}/models")
    os.makedirs(f"results/{path}/plots")
    os.makedirs(f"results/{path}/data")
logger.setLevel(param.level)
fileHandler = logging.FileHandler(f"results/{path}/logs/output.log", mode="w")
formatter = logging.Formatter(
    "%(asctime)s :: %(funcName)s :: %(levelname)s :: %(message)s"
)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

# define the network architecture


def initializeNetwork(params: dataclass) -> nn.Module:
    """
    Initialize the model
    Input:
        params: input parameters from the input file
    Output:
        model: torch.nn.Module
    """
    models = {"FNO3d": FNO3d, "MNO": FNO2d, "FNO": FNO}
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
    logger.debug(model)
    return model


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

    def batchLoop(self, train_loader, output_encoder=None):
        # initialize loss
        train_loss = 0
        loss = 0
        diss_loss = 0
        diss_loss_l2 = 0

        for batch_idx, sample in enumerate(train_loader):
            data, target = sample["x"].to(self.device), sample["y"].to(self.device)
            logger.debug(f"Batch Data Shape: {data.shape}")

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(data)
            logger.debug(f"Output Data Shape: {output.shape}")

            # decode  if there is an output encoder
            if output_encoder is not None:
                # decode the target
                target = output_encoder.decode(target)
                # decode the output
                output = output_encoder.decode(output)

            # compute the loss
            loss = self.loss(output.float(), target)
            logger.debug(f"Loss Shape: {loss.shape}")
            logger.debug(f"Loss Type: {type(loss)}")
            logger.debug(f"Loss Value: {loss}")
            logger.debug(f"Loss Value: {loss.item()}")
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
            test_loss_fn = LpLoss(d=2, p=2, reduce_dims=(0, 1))
        else:
            test_loss_fn = self.loss
        # start training
        for epoch in range(self.params.epochs):
            epoch_timer = time.time()

            # set to train mode
            self.model.train()
            train_loss, loss = self.batchLoop(train_loader,output_encoder=output_encoder)


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
        if self.params.saveNeuralNetwork:
            torch.save(self.model, f"results/{path}/models/{self.params.NN}")

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
        import matplotlib.pyplot as plt
        import seaborn as sns

        # set the style
        sns.set_style("darkgrid")

        ############ RMSE ############
        # compute RMSE
        rmse = torch.sqrt(torch.mean((prediction - truth) ** 2, dim=(1, 2, 3)))
        # normalize RMSE
        rmse /= torch.sqrt(torch.mean(truth**2, dim=(1, 2, 3)))
        # compute RMSE for each time step
        relativeRMSE = torch.sqrt(torch.mean((input - truth) ** 2, dim=(1, 2, 3)))
        # normalize RMSE
        relativeRMSE /= torch.sqrt(torch.mean(truth**2, dim=(1, 2, 3)))

        try:
            time = timeData["t"]["time"].values
        except:
            time = np.arange(0, prediction.shape[0])
            logger.info("Failed true time data")
        # plot RMSE for each time step
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(time, rmse, label="Neural RMSE")
        ax.plot(time, relativeRMSE, label="RMSE between time steps")
        ax.set_xlabel("Time (Myr)")
        ax.set_ylabel("Normalized  RMSE")
        ax.set_title("RMSE Over Time")
        ax.legend()
        im = wandb.Image(fig)
        wandb.log({"RMSE": im})
        if len(savename) > 0:
            fig.savefig(f"results/{path}/plots/{savename}_RMSE.png")
        plt.close(fig)

        ############ RMSE Vs Density ############
        # compute the RMSE as a function of density
        predBins, predMean, predSTD = self.densityRMSE(prediction, truth)
        inputBins, inputMean, inputSTD = self.densityRMSE(input, truth)
        # plot the RMSE as a function of density
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.loglog(predBins[:-1], predMean, label="Neural RMSE")
        ax.fill_between(
            predBins[:-1], predMean - predSTD, predMean + predSTD, alpha=0.2
        )
        ax.loglog(inputBins[:-1], inputMean, label="RMSE between time steps")
        ax.fill_between(
            inputBins[:-1], inputMean - inputSTD, inputMean + inputSTD, alpha=0.2
        )
        ax.set_xlabel(r"Column Density $  \left[\frac{g}{cm^2} \right]$")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.set_title("RMSE as a function of density")
        im = wandb.Image(fig)
        wandb.log({"RMSE vs Density": im})
        if len(savename) > 0:
            fig.savefig(f"results/{path}/plots/{savename}_RMSE_vs_density.png")
        plt.close(fig)

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
        pred = torch.zeros((num_samples, self.params.T, self.params.S, self.params.S))
        rePred = torch.zeros((num_samples, self.params.T, self.params.S, self.params.S))
        input = torch.zeros((num_samples, self.params.T, self.params.S, self.params.S))
        truth = torch.zeros((num_samples, self.params.T, self.params.S, self.params.S))
        test_loss = 0
        re_loss = 0

        with torch.no_grad():
            for batchidx, sample in enumerate(data_loader):
                data, target = sample["x"].to(self.device), sample["y"].to(self.device)
                if batchidx > 0:
                    if input_encoder:
                        reData = input_encoder.encode(reData)
                    # apply the model to previous output
                    reData = self.model(reData)
                output = self.model(data)
                if batchidx == 0:
                    reData = output.clone()
                # decode  if there is an output encoder
                if output_encoder:
                    # decode the target
                    # target = output_encoder.decode(target)
                    # decode the output
                    output = output_encoder.decode(output)
                    # decode the multiple applications of the model
                    reData = output_encoder.decode(reData)
                re_loss += self.loss(reData, target).item()
                test_loss += self.loss(output, target).item()
                if input_encoder:
                    data = input_encoder.decode(data)
                # assign the values
                # index = batchidx * self.params.batch_size
                # pred[index : index + self.params.batch_size] = output
                # input[index : index + self.params.batch_size] = data
                # truth[index : index + self.params.batch_size] = target
                pred[batchidx : batchidx + 1] = output
                input[batchidx : batchidx + 1] = data
                truth[batchidx : batchidx + 1] = target
                if batchidx > 0:
                    rePred[batchidx : batchidx + 1] = reData
                else:
                    rePred[batchidx : batchidx + 1] = output
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
