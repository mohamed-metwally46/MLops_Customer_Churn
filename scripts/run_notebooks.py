"""Execute notebooks with this interpreter; save local outputs outside tracked sources."""

import argparse
import json
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verification",
        action="store_true",
        help="Replay pinned saved results without training; preserve prior outputs.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    jupyter = root / ".venv" / "share" / "jupyter"
    kernels = jupyter / "kernels"
    kernel_dir = kernels / "churn"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Python 3 (churn)",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    os.environ["JUPYTER_PATH"] = str(jupyter)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(root / ".venv" / "jupyter-runtime")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(root / ".venv" / "matplotlib")
    os.environ["IPYTHONDIR"] = str(root / ".venv" / "ipython")
    output = root / "notebooks" / "executed"
    if args.verification:
        label = os.getenv("CHURN_VERIFICATION_LABEL", "milestone4")
        if not label.isidentifier():
            raise ValueError("Verification label must be a simple identifier.")
        output = output / label
        os.environ["CHURN_REPLAY_REPORT"] = str(
            root / ("reports/milestone3/a644309446d34b7ca25ea160124aeba6/results.json")
        )
        os.environ["CHURN_EDA_REPORT_DIR"] = f"reports/{label}/eda"
    output.mkdir(parents=True, exist_ok=True)
    for source in sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(
            notebook, timeout=1200, kernel_name="churn", resources={"metadata": {"path": str(root)}}
        )
        client.km = AsyncKernelManager(
            kernel_name="churn", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)])
        )
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output / source.name)
        print(f"Executed {source.name} -> {output.relative_to(root) / source.name}", flush=True)


if __name__ == "__main__":
    main()
