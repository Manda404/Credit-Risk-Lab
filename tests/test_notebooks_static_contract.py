import ast
import json
from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
FORBIDDEN_SNIPPETS = (
    "sys.path",
    "MPLCONFIGDIR",
    "run_source_quality_workflow(",
    "run_split_and_drift_workflow(",
    "run_feature_engineering_workflow(",
    "run_training_workflow(",
    "run_external_evaluation_workflow(",
)


def notebook_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_notebooks_do_not_hide_steps_behind_large_workflows():
    for path in NOTEBOOKS_DIR.glob("*.ipynb"):
        source = "\n".join(notebook_code_cells(path))
        for snippet in FORBIDDEN_SNIPPETS:
            assert snippet not in source, f"{snippet!r} found in {path.name}"


def test_notebook_code_cells_are_syntactically_valid():
    for path in NOTEBOOKS_DIR.glob("*.ipynb"):
        for cell_source in notebook_code_cells(path):
            ast.parse(cell_source)


def test_notebooks_import_package_components_explicitly():
    expected_imports = {
        "CsvLoanDataLoader",
        "DatasetInspector",
        "DataLeakageAuditor",
        "CreditRiskQualityChecker",
        "DatasetSplitter",
        "DriftAnalyzer",
        "LoanFeatureEngineer",
        "CreditRiskPreprocessor",
        "BoostingModelTrainer",
        "CreditRiskModelEvaluator",
        "JoblibModelBundleRepository",
        "RawLoanScorer",
        "CreditRiskMLOpsPipeline",
    }
    source = "\n".join(
        cell
        for path in NOTEBOOKS_DIR.glob("*.ipynb")
        for cell in notebook_code_cells(path)
    )
    for component in expected_imports:
        assert component in source
