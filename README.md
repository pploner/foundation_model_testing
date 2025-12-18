<div align="center">

# Repository for loading and processing the COLLIDE-2V dataset and training for a foundation model

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

At its current state, this repository is loading data from the COLLIDE-2V dataset as saved on EOS, saving it in a vectorized form usable for training, preprocessing it, and using it to train either a tinyMLP or a tinyTransformer classifier. The detailled and modular Hydra config allows for detailled configuration of feature and processes selection, preprocessing and standardization methods and model and training parameters. The code can be readily used to train classifiers with the COLLIDE-2V dataset. 

The repository follows the lightning-hydra framework that is accessible here: <https://github.com/ashleve/lightning-hydra-template> . For more details on the structure of the code please consult the readme there. 

Configuration of all aspects of the workflow is handled in the .yaml files in `configs/`, where more detailled documentations on the various config parameters can also be found. 

The repository is build with support for the Mlflow logger and hyperparameter optimization with Optuna.

Please send any feedback or suggestions to plonerp@ethz.ch.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/pploner/foundation_model_testing.git
cd foundation_model_testing

# build container with dependencies
apptainer build fm_testing.sif fm_testing.def
```

If the dataset itself has been updated, the event map at `src/utils/nEvents_scan/file_event_counts.json` will need to be re-generated. This can be done by executing the `scan_parquet_nevent.py` script.

## Usage

### Configuration
The configuration of every aspect of the pipeline is handled in the Hydra .yaml files in `configs`. The optimal workflow is to fix all the parameters you don't change inside the respective files, and only overwrite the ones you change in a corresponding experiment file that is then called in `configs/train.yaml`. Documentation is available as comments inside the config files. 

### Full Pipeline
The full vectorization-preprocessing-training-evaluation pipeline can be executed by calling `python src/train.py`. Vectorized and preprocessed data will be saved in corresponding .npy files inside the directories given in `configs/paths` together with a feature map file that maps the dataset features to their position in the .npy files. 

If the corresponding .npy files already exist, the script will skip those steps and move directly to training. 

Model weights are saved in checkpoint files in the folder of the corresponding run in `logs/`. If one wants to perform just model evaluation using a given checkpoint, one can instead use `src/eval.py`.

### Separate Vectorization and Preprocessing
Since the dataset size is enormous, depending on the feature selection and sample size the vectorization and preprocessing procedures might take on the order of weeks. For this purpose, vectorization and preprocessing can be performed independently first, using the HTCondor batch submission framework on Lxplus. `scripts/submit_vectorization_jobs.py` and `scripts/submit_preprocessing_jobs.py` can be used to do those steps in an optimized manner, where the tasks are equally distributed over many compute nodes, saving a lot of time. Take care to only submit the preprocessing once all vectorization jobs have finished.

### Data Validation
The features of the source dataset .parquet files can be inspected with `scripts/parquet_plotter.py`, and the features of the vectorization and preprocessing .npy files can be plotted using `scripts/plot_features.py`. Use these scripts to validate and inspect the data.

### Training Iterations and GPU Usage
If all the preprocessed files are ready, training can simply again be done with `src/train.py`, which will recognize the available files and skip vectorization and preprocessing. Training can also be submitted to compute nodes using the condor batch submission system via `condor_submit src/train_full_pipeline.sub`. Given the corresponding trainer setting, training on GPU is also supported locally and via condor submission. All condor submission logs will by default be saved in `logs/condor_logs`.

#### Supported Models
The currently implemented models are an MLP and a Transformer. Their Lightning source code is available in `src/models`. More custom models can be implemented too and should follow the Lightning framework to be compatible with the rest of the repository.

### Logging and Hyperparameter Sweeps
The repository is build with support of the mlflow logger and the optuna hyperparameter sweeper. Logs of your runs can be accessed on the mlflow user interface by calling `mlflow ui` from inside `logs/mlflow`. If you perform a hyperparameter sweep, its output will be saved in a .db file that can be inspected using `notebooks/optuna_sweep_results.ipynb`.

