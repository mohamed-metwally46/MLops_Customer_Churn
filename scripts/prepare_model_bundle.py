"""Prepare a local, ignored two-file model mount; never publish or alter originals."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from churn.data import CATEGORIES, FEATURES, SEED
from churn.model_loader import PINNED_REPORT, ModelSettings, load_model
from churn.preprocessing import preprocessing


def prepare_bundle(output: Path, root: Path, synthetic: bool = False) -> dict:
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing bundles are preserved.")
    if synthetic:
        # CI-only fixture: no dataset acquisition, private data or model selection.
        frame = pd.DataFrame(
            {
                c: [sorted(values, key=str)[i % len(values)] for i in range(60)]
                for c, values in CATEGORIES.items()
            }
        )
        frame["tenure"] = np.arange(60)
        frame["MonthlyCharges"] = 20.0 + np.arange(60)
        frame["TotalCharges"] = frame["tenure"] * frame["MonthlyCharges"]
        model = Pipeline(
            [
                ("preprocess", preprocessing()),
                ("select", "passthrough"),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=10, max_depth=3, random_state=SEED, n_jobs=1
                    ),
                ),
            ]
        )
        model.fit(frame[FEATURES], (np.arange(60) % 3 == 0).astype(int))
        metadata = {
            "final_run_id": hashlib.sha256(b"synthetic-ci-fixture-v1").hexdigest()[:32],
            "frozen_decision": {"model_type": "RandomForest", "threshold": 0.5},
        }
        output.mkdir(parents=True)
        joblib.dump(model, output / "pipeline.joblib")
    else:
        metadata = json.loads((root / PINNED_REPORT).read_text(encoding="utf-8"))
        load_model(ModelSettings(root, root / PINNED_REPORT))
        data = (root / metadata["artifact"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != metadata["artifact_sha256"]:
            raise ValueError("Trusted artifact fingerprint mismatch; no bundle written.")
        output.mkdir(parents=True)
        (output / "pipeline.joblib").write_bytes(data)
    compact = {
        "artifact": "pipeline.joblib",
        "input_features": FEATURES,
        "artifact_sha256": hashlib.sha256((output / "pipeline.joblib").read_bytes()).hexdigest(),
        "final_run_id": metadata["final_run_id"],
        "frozen_decision": metadata["frozen_decision"],
        "synthetic_ci_only": synthetic,
    }
    (output / "metadata.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    output.chmod(0o755)
    for path in output.iterdir():
        path.chmod(0o644)
    load_model(ModelSettings(output, output / "metadata.json"))
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true", help="CI fixture, not a release model")
    args = parser.parse_args()
    result = prepare_bundle(args.output.resolve(), Path.cwd(), args.synthetic)
    print("Local bundle verified:", result["final_run_id"], "synthetic:", args.synthetic)


if __name__ == "__main__":
    main()
