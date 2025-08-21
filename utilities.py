import os
import mlflow
from ultralytics import YOLO
import tempfile

domino_user_name = os.environ['DOMINO_USER_NAME']
model_registration_experiment_name=f"model_registration-{domino_user_name}"
cv_comparison_experiment_name=f"cv-comparison-{domino_user_name}"

def ensure_mlflow_experiment(experiment_name: str) -> int:
    """
    Ensure an MLflow experiment exists and set it as current.

    If an experiment with `experiment_name` does not exist, create it. In both cases,
    set the active experiment so subsequent runs attach correctly.

    Parameters
    ----------
    experiment_name : str
        The MLflow experiment name.

    Returns
    -------
    str
        The experiment ID.

    Raises
    ------
    RuntimeError
        If the experiment lookup/creation fails.

    Notes
    -----
    - The MLflow tracking URI and token are pre-configured in Domino
    """
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(
                experiment_name
            )
        else:
            exp_id = exp.experiment_id
        mlflow.set_experiment(experiment_name)
        return exp_id
    except Exception as e:
        raise RuntimeError(f"Failed to ensure experiment {experiment_name}: {e}")
        


def load_registered_yolo_model(model_name: str, version: str = "latest"):
    """
    Resolve a registered ONNX model from MLflow and load it into Ultralytics YOLO.

    The function downloads the specified registered model version to a unique directory
    under `/tmp` and constructs a `YOLO` instance pointing at the resolved `.onnx` file.

    Parameters
    ----------
    model_name : str
        Registered model name in MLflow Model Registry (e.g., "yolov8n").
    version : str, optional
        Registered model version identifier. Use "latest" to resolve the latest version.
        Otherwise pass a numeric string like "3". Default is "latest".

    Returns
    -------
    ultralytics.YOLO
        A YOLO object ready for inference/validation, configured with `task="detect"`.

    Raises
    ------
    FileNotFoundError
        If no `.onnx` file is found in the downloaded model directory.
    mlflow.exceptions.MlflowException
        If MLflow cannot resolve or download the requested model artifacts.

    Notes
    -----
    - Artifact layout is flavor-dependent. This function scans the downloaded directory
      recursively for a single `.onnx` file and loads the first match.
    - To avoid `/dev/shm` issues during evaluation in constrained environments, run
      `model.val(..., workers=0)`.
    - The HW Tier is configured to use shared memory as high as 10GB
    """
    if version == "latest":
        model_uri = f"models:/{model_name}/latest"
    else:
        model_uri = f"models:/{model_name}/{version}"

    # Create a unique temp dir under /tmp
    tmp_dir = tempfile.mkdtemp(prefix=f"{model_name}_", dir="/tmp")

    # Download artifacts into /tmp
    local_dir = mlflow.artifacts.download_artifacts(model_uri, dst_path=tmp_dir)

    # Find ONNX file
    onnx_candidates = []
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith(".onnx"):
                onnx_candidates.append(os.path.join(root, f))

    if not onnx_candidates:
        raise FileNotFoundError(f"No .onnx file found under {local_dir}")

    onnx_path = onnx_candidates[0]
    print(f"[Loaded] {model_name}:{version} from {onnx_path}")

    return YOLO(onnx_path, task="detect")
