import numpy as np
import pandas as pd

from credit_risk_lab.application import TrainBoostingModelsUseCase


def synthetic_modeling_frame(rows: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.uniform(20_000, 120_000, rows)
    credit_score = rng.uniform(400, 800, rows)
    target = (credit_score < 600).astype(int)
    return pd.DataFrame(
        {
            "person_income": income,
            "credit_score": credit_score,
            "dti": rng.uniform(0, 1, rows),
            "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], rows),
            "person_gender": rng.choice(["male", "female"], rows),
            "is_female": rng.integers(0, 2, rows),
            "loan_status": target,
        }
    )


def test_sensitive_columns_are_excluded_from_training_features():
    frame = synthetic_modeling_frame()
    use_case = TrainBoostingModelsUseCase(n_estimators=10)
    result = use_case.execute(frame)

    trained_feature_names = list(result.preprocessor.get_feature_names_out())
    assert "person_gender" not in trained_feature_names
    assert "is_female" not in trained_feature_names
    for column in trained_feature_names:
        assert not column.startswith("person_gender_")

    assert {"person_gender", "is_female"}.issubset(result.sensitive_test.columns)
    assert len(result.sensitive_test) > 0
