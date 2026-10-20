import os
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow


class MLflowTracker:
    """
    LLMOps Experiment Tracking & Model Governance via MLflow.
    Uses SQLite database backend ('sqlite:///mlflow.db') for production-ready tracking.
    Tracks:
    - RAG retrieval hyperparameters (chunk_size, chunk_overlap, k, score_threshold)
    - LLM generation parameters (model, temperature, prompt template versions)
    - Live inference metrics (latency, retrieval depth, tool selection)
    - RAGAS evaluation benchmark runs (faithfulness, context relevance, answer relevance)
    - System artifacts (prompt configs, benchmark results datasets, traces)
    """

    def __init__(
        self,
        experiment_name: str = "VidSense-AI-Agent",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        # Use a local SQLite database backend inside backend/mlflow.db
        if tracking_uri is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            db_path = (base_dir / "mlflow.db").as_posix()
            self.tracking_uri = f"sqlite:///{db_path}"
        else:
            self.tracking_uri = tracking_uri

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self.enabled = True
        except Exception as e:
            print(f"[MLflowTracker] Warning: MLflow initialization failed: {e}")
            self.enabled = False

    def log_inference_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ):
        """Logs a single inference trace run into MLflow."""
        if not self.enabled:
            return None

        try:
            with mlflow.start_run(run_name=run_name, nested=True):
                if tags:
                    mlflow.set_tags(tags)
                if params:
                    safe_params = {k: str(v)[:250] for k, v in params.items()}
                    mlflow.log_params(safe_params)
                if metrics:
                    mlflow.log_metrics(metrics)
                if artifacts:
                    for art_path in artifacts.values():
                        if os.path.exists(art_path):
                            mlflow.log_artifact(art_path)
        except Exception as e:
            print(f"[MLflowTracker] Error logging inference run: {e}")

    def log_evaluation_run(
        self,
        run_name: str,
        eval_metrics: Dict[str, float],
        hyperparams: Dict[str, Any],
        dataset_path: Optional[str] = None,
        results_csv_path: Optional[str] = None,
    ):
        """Logs a RAGAS benchmark evaluation experiment into MLflow."""
        if not self.enabled:
            return None

        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({
                    "stage": "evaluation",
                    "framework": "RAGAS",
                    "vector_store": "ChromaDB",
                })
                if hyperparams:
                    mlflow.log_params({k: str(v) for k, v in hyperparams.items()})
                if eval_metrics:
                    mlflow.log_metrics(eval_metrics)
                if dataset_path and os.path.exists(dataset_path):
                    mlflow.log_artifact(dataset_path, artifact_path="datasets")
                if results_csv_path and os.path.exists(results_csv_path):
                    mlflow.log_artifact(results_csv_path, artifact_path="evaluation_reports")
                print(f"[MLflowTracker] Evaluation logged successfully to run '{run_name}'.")
        except Exception as e:
            print(f"[MLflowTracker] Error logging evaluation run: {e}")
