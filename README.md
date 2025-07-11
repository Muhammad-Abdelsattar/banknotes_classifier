# Banknote Classifier

![Status](https://img.shields.io/badge/status-work_in_progress-yellow) ![Python](https://img.shields.io/badge/python-3.10-blue.svg) ![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![DVC](https://img.shields.io/badge/DVC-DataVersionControl-blue?logo=dvc) ![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-blue?logo=githubactions) ![License](https://img.shields.io/badge/License-Not%20Specified-lightgrey)

This project implements a machine learning model to classify images of banknotes, primarily using synthetically generated data for training. It includes a DVC pipeline for data processing, model training, and evaluation, as well as CI/CD workflows for automated experiment tracking and deployment of a demo application.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data](#data)
- [Modeling](#modeling)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Export](#model-export)
- [DVC Pipeline](#dvc-pipeline)
- [CI/CD and MLOps](#cicd-and-mlops)
- [Demo Application](#demo-application)
- [How to Contribute](#how-to-contribute)
- [License](#license)

## Project Structure

The project is organized as follows:

-   `banknotes_classifier/`: Contains the core Python package for the banknote classifier.
    -   `data/`: Modules for data loading, preprocessing, and augmentation.
    -   `modeling/`: Defines the model architecture and PyTorch Lightning module.
    -   `training/`: Scripts and utilities for model training.
    -   `evaluation/`: Scripts for model evaluation and inference pipelines.
-   `deployment/`: Contains application code, demo code, and Docker configuration.
    -   `app/`: Contains the code for the production inference application (FastAPI).
    -   `demo_app/`: Contains the code for the Gradio demo application.
    -   `Dockerfile`: Dockerfile for building the production application image.
-   `input/`: Stores raw datasets and DVC metadata files for versioning large data files.
    -   `dataset.zip.dvc`: DVC file for the training dataset.
    -   `test_dataset.zip.dvc`: DVC file for the test dataset.
-   `artifacts/`: Stores DVC-tracked model checkpoints and exported model files.
    -   `checkpoints/`: Saved model checkpoints during training.
    -   `model.onnx`: The model exported to ONNX format.
-   `reports/`: Stores DVC-tracked reports, including training and evaluation metrics and plots.
    -   `training/`: Metrics and plots generated during training.
    -   `evaluation/`: Metrics (e.g., `metrics.json`) and plots (e.g., `confusion_matrix.png`) generated during evaluation.
-   `scripts/`: Contains utility scripts, such as `dataset_generation.py` for creating or preparing datasets.
-   `steps/`: Contains Python scripts that define individual stages in the DVC pipeline (e.g., `train.py`, `evaluate.py`, `export.py`).
-   `.github/workflows/`: Contains GitHub Actions workflow files for CI/CD and MLOps automation.
    -   `test_exp_pr_dev.yaml`: Workflow for testing experiments, tracking metrics, versioning models, and creating PRs.
    -   `deploy_demo_app.yaml`: Workflow for deploying the demo application to Hugging Face Spaces.
-   `dvc.yaml`: Defines the DVC pipeline stages, dependencies, and outputs.
-   `params.yaml`: Configuration file for data, training, and evaluation parameters.
-   `requirements.txt`: Lists the Python dependencies for the project.
-   `README.md`: This file.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

-   Python (version 3.10 or higher recommended, as per workflow files)
-   DVC (Data Version Control): [Install DVC](https://dvc.org/doc/install)
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Muhammad-Abdelsattar/banknotes_classifier
    cd banknotes_classifier
    ```

2.  **Install Python dependencies:**
    It's recommended to create a virtual environment first.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up DVC and pull data/models:**
    You'll need to configure the DVC remote to access the data. The project uses Google Drive. The following steps are based on the CI workflows:
    ```bash
    # This step might require credentials (e.g., creds.json from Google Cloud service account)
    # The CI uses a secret GDRIVE_CREDENTIALS_DATA. For local setup, you might need to configure this manually.
    # Refer to DVC documentation for Google Drive remote setup: https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive
    
    # Example based on CI (you may need to adapt this):
    # dvc remote add -d myremote gdrive://<your_gdrive_folder_id> 
    # dvc remote modify myremote gdrive_use_service_account true 
    # dvc remote modify myremote gdrive_service_account_json_file_path path/to/your/creds.json

    # Once the remote is configured, pull the DVC-tracked files:
    dvc pull
    ```
    This command will download the datasets from `input/` and the latest model artifacts from `artifacts/` as defined in their respective `.dvc` files and `dvc.lock`.

## Data

The primary dataset used for training the banknote classifier is **synthetically generated**. This project also manages test data using DVC.

-   **Training Data (Synthetic)**: The training dataset is generated by the `scripts/dataset_generation.py` tool. The DVC-tracked file at `input/dataset.zip` (and its corresponding `.dvc` file) represents an instance of such a synthetically generated dataset. This archive is unzipped into `input/dataset/` during the `prepare` stage of the DVC pipeline.
-   **Test Data**: Located at `input/test_dataset.zip` (tracked by `input/test_dataset.zip.dvc`). This archive contains real or separately prepared images for testing and is unzipped into `input/test_dataset/` during the `prepare` stage.

The DVC-tracked data files are stored in a Google Drive remote, configured as per the DVC setup.

To download the DVC-tracked data versions:
```bash
dvc pull
# This will pull input/dataset.zip and input/test_dataset.zip among other DVC-tracked files.
# The prepare stage (dvc repro prepare) will then unzip them.
```

The script `scripts/dataset_generation.py` is crucial as it provides the functionality to create the synthetic training dataset. To use it, you need to provide paths to two main input directories:
-   One directory containing the original banknote images, typically organized into subdirectories by class (e.g., `banknotes/`, passed to the `banknotes_path` parameter).
-   Another directory containing various background images (e.g., `backgrounds/`, passed to the `backgrounds_path` parameter).
The script then takes these original banknotes, creates numerous variations (rotations, halves using its `make_variations` function into an intermediate `banknotes_vars_path`), applies a series of augmentations (e.g., brightness, contrast, perspective changes), and overlays them onto the background images. This process significantly expands the dataset volume and diversity, which is essential for training a robust model. The output is saved to a specified `dataset_path`.

## Modeling

The core logic for the banknote classification model resides within the `banknotes_classifier/modeling/` directory.

-   **Model Architectures (`model.py`)**: This file defines a `BaseClassifier` class and several specific convolutional neural network architectures that inherit from it, including:
    -   `MobileNetClassifier`
    -   `Regnet400Classifier`
    -   `EfficientNetClassifier` (used in the current DVC pipeline for training and export)
    It also contains an `ExportReadyModel` class, which wraps a trained model (like `EfficientNetClassifier`) to include image normalization and a Softmax activation for consistent output during inference, especially for the ONNX exported model.

-   **PyTorch Lightning Module (`lit_module.py`)**: The `BanknotesClassifierModule` is a `pytorch_lightning.LightningModule` that orchestrates the training process. It encapsulates:
    -   The chosen model (e.g., `EfficientNetClassifier`).
    -   The optimization algorithm (AdamW).
    -   The loss function (CrossEntropyLoss).
    -   The training, validation steps, and metric calculations (accuracy).
    -   Data loading via PyTorch DataLoaders.
    This modular approach simplifies training and ensures compatibility with PyTorch Lightning's advanced features.

Key configurable parameters related to the model and training process can be found in `params.yaml`.

## Training

Model training is handled by the `steps/train.py` script, which is executed as part of the DVC pipeline.

To initiate model training:

1.  Ensure your data is downloaded and prepared (see [Data](#data) and [Getting Started](#getting-started)).
2.  Run the training stage using DVC:
    ```bash
    dvc repro train
    ```
    Alternatively, you can directly execute the training script (though using `dvc repro` is recommended to ensure all dependencies are up to date):
    ```bash
    python -m steps.train
    ```

Key training parameters are defined in `params.yaml` under the `training` section. These include:
-   `batch_size`
-   `lr` (learning rate)
-   `trainer`: Configuration for the PyTorch Lightning `Trainer` (e.g., `accelerator`, `max_epochs`).
-   `callbacks`: Configuration for callbacks like `ModelCheckpoint` (for saving the best model), `ModelSummary`, and `TQDMProgressBar`.
-   `logger`: Configuration for logging training progress, with outputs typically saved in `reports/training/`.

The best model checkpoint, based on the `valid_loss` metric, is saved to `artifacts/checkpoints/best-checkpoint.ckpt`. Training metrics are saved in `reports/training/metrics.json`.

## Evaluation

Model evaluation is performed by the `steps/evaluate.py` script, which is executed as part of the DVC pipeline. This step typically uses the exported ONNX model.

To evaluate the model:

1.  Ensure the model has been trained and exported (or download a pre-trained model using DVC).
2.  Ensure the test dataset (`input/test_dataset/`) is available.
3.  Run the evaluation stage using DVC:
    ```bash
    dvc repro evaluate
    ```
    Or directly (not recommended for pipeline consistency):
    ```bash
    python -m steps.evaluate
    ```

Evaluation parameters, such as the path to the test dataset, are specified in `params.yaml` under the `evaluation` section.

The evaluation process generates:
-   **Metrics**: Saved in `reports/evaluation/metrics.json`. These metrics are crucial for assessing model performance.
-   **Plots**: Such as a confusion matrix, saved in `reports/evaluation/plots/confusion_matrix.png`. These plots provide visual insights into the model's behavior.

These outputs are versioned by DVC and are used in the CI/CD workflow to compare performance between experiments.

## Model Export

The trained model can be exported to the ONNX (Open Neural Network Exchange) format, which allows for interoperability between different deep learning frameworks and tools. The export process is handled by the `steps/export.py` script.

To export the model:

1.  Ensure that a trained model checkpoint exists (e.g., `artifacts/checkpoints/best-checkpoint.ckpt`). This is created during the [Training](#training) process.
2.  Run the export stage using DVC:
    ```bash
    dvc repro export
    ```
    Alternatively, you can execute the export script directly:
    ```bash
    python -m steps.export
    ```

The script loads the best checkpoint specified in `params.yaml` (under `training.callbacks.ModelCheckpoint`) and uses the `EfficientNetClassifier` model architecture (defined in `banknotes_classifier/modeling/model.py`). The exported model is saved as `artifacts/model.onnx`, as configured in `params.yaml` (`export.model_path`).

## DVC Pipeline

This project uses DVC (Data Version Control) to manage the ML experimentation pipeline, ensuring reproducibility and efficient data handling. The pipeline is defined in `dvc.yaml` and consists of several stages:

1.  **`prepare`**:
    *   **Command**: `unzip -o input/dataset.zip -d input/dataset > /dev/null 2>&1 && unzip -o input/test_dataset.zip -d input/test_dataset > /dev/null 2>&1`
    *   **Description**: Unzips the raw training (`dataset.zip`) and testing (`test_dataset.zip`) datasets into their respective directories (`input/dataset` and `input/test_dataset`).
    *   **Dependencies**: `input/dataset.zip`, `input/test_dataset.zip`.
    *   **Outputs**: `input/dataset`, `input/test_dataset`.

2.  **`train`**:
    *   **Command**: `python -m steps.train`
    *   **Description**: Runs the model training script.
    *   **Dependencies**: `steps/train.py`, `banknotes_classifier/modeling/`, `banknotes_classifier/training/`, `banknotes_classifier/data/`, `input/dataset` (prepared data).
    *   **Outputs**: 
        *   `artifacts/checkpoints/best-checkpoint.ckpt` (the best model checkpoint).
        *   `reports/training/metrics.json` (training metrics, not cached by DVC).

3.  **`export`**:
    *   **Command**: `python -m steps.export`
    *   **Description**: Exports the trained model to ONNX format.
    *   **Dependencies**: `steps/export.py`, `artifacts/checkpoints/best-checkpoint.ckpt`.
    *   **Outputs**: `artifacts/model.onnx` (the exported ONNX model).

4.  **`evaluate`**:
    *   **Command**: `python -m steps.evaluate`
    *   **Description**: Runs the model evaluation script using the exported ONNX model and the test dataset.
    *   **Dependencies**: `steps/evaluate.py`, `artifacts/model.onnx`, `input/test_dataset`.
    *   **Outputs**:
        *   `reports/evaluation/metrics.json` (evaluation metrics, not cached by DVC, note: original dvc.yaml has a typo `reports/evaluaton/metrics.json`, using corrected version here).
        *   `reports/evaluation/plots/confusion_matrix.png` (confusion matrix plot, not cached by DVC).

**Running the Pipeline:**

To run the entire pipeline from start to finish, ensuring all stages are executed in the correct order and only if their dependencies have changed:
```bash
dvc repro
```

To run a specific stage and its dependencies:
```bash
dvc repro <stage_name> 
# e.g., dvc repro train
```

The `params.yaml` file is also listed as a global parameter dependency, meaning changes to it can trigger re-runs of relevant stages. DVC also tracks metrics and plots generated by the pipeline, as defined under the `metrics` and `plots` sections in `dvc.yaml`.

## CI/CD and MLOps

The project incorporates CI/CD and MLOps practices using GitHub Actions to automate various parts of the development and deployment lifecycle. The workflows are defined in the `.github/workflows/` directory.

### Key Workflows:

1.  **Experiment Tracking and Validation (`.github/workflows/test_exp_pr_dev.yaml`)**
    *   **Trigger**: Pushes to branches matching the pattern `exp/*` (e.g., `exp/my-new-feature`).
    *   **Purpose**: To automate the evaluation of new experiments, track their performance, and streamline their promotion if successful. PRs are created to `dev` and experiment tags like `myexp-v1` are pushed. (This experiment tag serves as a baseline for features but is not directly transformed into staging/production tags anymore.)
    *   **Process**:
        1.  Checks out the code and sets up the Python environment on a self-hosted runner.
        2.  Installs project dependencies.
        3.  Configures DVC to use a Google Drive remote for accessing data and models.
        4.  Pulls the latest production model (`artifacts/model.onnx`) and the test dataset (`input/test_dataset.zip`) using DVC.
        5.  Runs the evaluation script (`python -m steps.evaluate`) to generate metrics for the current experiment.
        6.  Commits and pushes any changes to metrics files (`reports/evaluation/metrics.json`) and plots (`reports/evaluation/plots/confusion_matrix.png`) back to the experiment branch.
        7.  **Metric Comparison**: Compares the newly generated metrics against those from the `master` branch using `dvc metrics diff`. A helper script (`.github/workflows/helpers/check.py`) determines if the performance change is "acceptable."
        8.  **Experiment Registration (Tagging)**: If the metrics are deemed acceptable, a new Git tag is created for the experiment (e.g., `experiment_name-v0`, `experiment_name-v1`). This versions the successful experiment.
        9.  **Pull Request Creation**: A Pull Request is automatically created (or updated) from the experiment branch to the `dev` branch, signaling that the experiment is ready for review and integration.
        10. **PR Reporting**: A comment is added to the PR, including a report of the DVC metrics diff and the confusion matrix image, providing a clear summary of the experiment's performance.

2.  **Staging Tag Creation (`.github/workflows/create_staging_tag.yaml`)**
    *   **Trigger**: Pull Requests from `dev` branch to `master` branch (on opened or synchronized).
    *   **Purpose**: To create an auto-incremented 'staging' tag (e.g., `staging-v1`, `staging-v2`) for the codebase state intended for staging.
    *   **Process**:
        1.  Checks out the code from the PR's head commit (from `dev`).
        2.  Fetches all existing tags to find the latest `staging-vN` tag.
        3.  If no such tag exists, it defaults to `staging-v1`.
        4.  Increments the version number (e.g., `staging-v1` becomes `staging-v2`).
        5.  Pushes the new incremented staging tag (e.g., `staging-v2`) to the PR's head commit.

3.  **Production Release Publishing (`.github/workflows/publish_production_release.yaml`)**
    *   **Trigger**: Pushes to the `master` branch (typically after a PR from `dev` is merged).
    *   **Purpose**: To create an auto-incremented 'production' tag (e.g., `production-v1`, `production-v2`), build the Docker image, and publish it to Docker Hub.
    *   **Process**:
        1.  Checks out the code from the `master` branch.
        2.  Fetches all existing tags to find the latest `production-vN` tag.
        3.  If no such tag exists, it defaults to `production-v1`.
        4.  Increments the version number.
        5.  Pushes the new incremented production tag (e.g., `production-v2`) to the current commit on `master`.
        6.  Sets up Docker Buildx and logs into Docker Hub.
        7.  Extracts Docker metadata: the image is tagged with the new `production-vN` tag and `latest`.
        8.  Builds and pushes the Docker image from `deployment/app/` (using `deployment/Dockerfile`) to Docker Hub.

4.  **Demo Application Deployment (`.github/workflows/deploy_huggingface_demo.yaml`)**
    *   **Trigger**: When a Pull Request is opened or reopened targeting the `dev` branch.
    *   **Purpose**: To automatically deploy a demo version of the application to Hugging Face Spaces, allowing for easy testing and showcasing of the changes.
    *   **Process**:
        1.  Checks out the code and sets up the Python environment on a self-hosted runner.
        2.  Installs dependencies.
        3.  **Hugging Face Space Creation**: Creates a new (or uses an existing) Hugging Face Space. The Space name is dynamically generated based on the branch name and the latest version tag associated with it.
        4.  Configures DVC and pulls the required `model.onnx` artifact.
        5.  **Application Upload**: Copies `artifacts/model.onnx` into `deployment/demo_app/artifacts/` and then uploads the entire `deployment/demo_app/` contents to the designated Hugging Face Space.
        6.  **PR Comment**: Posts a comment on the Pull Request with a direct link to the deployed Hugging Face Space, allowing reviewers and stakeholders to interact with the demo.

### MLOps Practices Implemented:

*   **Version Control**: Git for code, DVC for data and models.
*   **Automated Testing & Evaluation**: CI pipeline automatically runs evaluation for every experiment.
*   **Experiment Tracking**: Metrics and parameters are tracked, and experiments are versioned using Git tags.
*   **Reproducibility**: DVC pipelines and versioned artifacts ensure that experiments and results can be reproduced.
*   **Continuous Integration**: Changes are frequently integrated and tested.
*   **Continuous Deployment (for Demos)**: Changes merged into `dev` (via PRs) trigger automatic deployment of a demo application.
*   **Collaboration**: Automated PR creation and reporting facilitate team collaboration and review.
*   **Automated Versioning for Staging and Production Tags**: Git tags like `staging-vN` (e.g., `staging-v1`) and `production-vN` (e.g., `production-v1`) are automatically created and incremented, providing clear versioning for these environments.
*   **Automated Production Docker Build & Push**: The application Docker image is automatically built and published to Docker Hub, versioned with the `production-vN` tag and `latest`.

## Applications and Deployment

This project provides multiple ways to interact with and deploy the banknote classifier:

1.  **Hugging Face Spaces Demo (Gradio)**
    *   **Location**: `deployment/demo_app/`
    *   **Description**: A web-based interactive demo built with Gradio (`deployment/demo_app/app.py`) for quick testing and visualization by uploading banknote images.
    *   **Deployment**: Automatically deployed to Hugging Face Spaces when a Pull Request is opened/reopened to the `dev` branch (see [CI/CD and MLOps](#cicd-and-mlops)).

2.  **Local API Application / Production Docker Image**
    *   **Location**: `deployment/app/`
    *   **Description**: This directory contains the core application (`deployment/app/main.py`, `deployment/app/inference.py`) that serves the classification model via a FastAPI-based API. This is the application packaged into a Docker image for production deployments.
    *   **Local Setup & Running**:
        *   Install dependencies: `pip install -r deployment/app/requirements.txt`
        *   Run (example for FastAPI with uvicorn):
            ```bash
            cd deployment/app
            uvicorn main:app --reload
            ```
    *   **Dockerization**: The `Dockerfile` is located in the `deployment/` directory, which is used to build the production image from the `deployment/app/` directory.

3.  **Docker Hub Deployment**
    *   **Status**: Implemented
    *   **Workflow**: `.github/workflows/publish_production_release.yaml`
    *   **Goal**: The Docker image from `deployment/app/` is automatically built and pushed to Docker Hub.
    *   **Trigger**: Pushes to the `master` branch.
    *   **Image Versioning**: Images are tagged with an auto-incremented version like `production-vN` (e.g., `production-v1`, `production-v2`) and `latest`. The image name on Docker Hub is `${{ secrets.DOCKERHUB_USERNAME }}/banknote_classifier`.

## How to Contribute

Contributions are welcome! If you'd like to contribute to this project, please follow these general steps:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix:
    *   For experimental features, consider using a naming convention like `exp/your-feature-name` to leverage the automated experiment tracking workflow.
    *   For other contributions, a descriptive branch name like `feature/new-optimizer` or `fix/readme-typo` is good.
3.  **Make your changes.** Ensure you add or update tests as appropriate.
    *   **Test your changes thoroughly.** If your changes affect model performance, the automated experiment tracking workflow (`.github/workflows/validate_experiment.yaml`) will help evaluate them if you push to an `exp/*` branch.
5.  **Ensure your code adheres to any existing coding standards.** (Linters or formatters might be used in the project - check for configuration files).
6.  **Commit your changes** with clear and descriptive commit messages.
7.  **Push your branch** to your forked repository.
8.  **Create a Pull Request** to the `dev` branch of the main repository. Provide a clear description of your changes in the PR.

If you are planning a major change, please open an issue first to discuss it with the maintainers.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
