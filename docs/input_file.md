# Input File

The input file is a python that specifies the configures the dataset, the neural network, and the training parameters. Here is an overview of the input file format.

<details>
    <summary>Example input.py</summary>

    ```python title="input.py"
    --8<-- "input.py"
    ```

</details>

## Dataset

The dataset is first specified by the variable `data_name` which specifies the dataset used.  This only specifies the type of data and further parameters are needed to specify the exact file to load.
The current options are:

- `'NS-Caltech'`
- `'StarForm'`
- `'GravColl'`
- `'CATS'`

### Variables for all datasets

The variables consistent for all datasets are shown in the table below. For typical values see the example input file.

| Variable |Type| Description | Requirements|
| --- | --- | --- | --|
| `DATA_PATH` | `str` | Path to the data |
| `TRAIN_PATH` | `str`| Path to the training data file |
| `TIME_PATH` | `str`| Path to the time data file |
| `log` | `bool` | Whether to take the log10 of the data. This is only applied to the density. |
| `S` | `int` |Size of the grid |
| `N` | `int` |Number of samples/trajectories |
| `input_channels`| `int` | Number of input channels for the neural operator |
| `output_channels`| `int` | Number of output channels for the neural operator |
| `modes` | `int` | Number of modes to keep in the neural operator |
| `width` | `int` | Width of the neural network |
| `T` | `int` | Number of time steps to predict |
| `T_in`| `int` | Number of time steps to use as input |
| `poolKernel` | `int` | Kernel size for average pooling. | Must be a divisor of of `S`|
| `poolStride` | `int` | Stride for pooling.| Must be a divisor of of `S`. Best to be the same value as `poolKernel`|

### Variables for specific datasets

| Variable |Type| Description | Dataset(s)|
| --- | --- | --- | --|
| `dN` | `int`|Number of timesteps per sample/trajectory  | `'StarForm'`, `'GravColl'`|
| `mu`| `str` | Relative magnetic strength to the gravitational force. Used for selecting which file to read. | `'StarForm'`|
| `mass`| `str` | Mass of the cloud in solar masses. Used for selecting which file to read. | `'GravColl'`|
| `dt`| `str` | Size of the timestep in kyr. Used for selecting which file to read. | `'GravColl'`|

## Neural Network
