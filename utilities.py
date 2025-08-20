import os
import mlflow
from ultralytics import YOLO
import tempfile

domino_user_name = os.environ['DOMINO_USER_NAME']
model_registration_experiment_name=f"model_registration-{domino_user_name}"
cv_comparison_experiment_name=f"cv-comparison-{domino_user_name}"

def ensure_mlflow_experiment(experiment_name: str) -> int:
    """
    Ensure an MLflow experiment with the given name exists.
    If it does not, create it. Then set it as the current experiment.

    Args:
        experiment_name: Name of the experiment
        artifact_location: Optional path or URI where artifacts will be stored

    Returns:
        experiment_id (int)
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
    Downloads the registered ONNX model version into /tmp and returns a Ultralytics YOLO instance.
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
