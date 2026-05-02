import pandas as pd
from spec4ml_studio.adapters.spec4ml_py_adapter import Spec4MLPyBackend
from spec4ml_studio.services.dataset_service import DatasetService
from spec4ml_studio.services.evaluation_service import EvaluationService
from spec4ml_studio.domain.models import DatasetSelection, EvaluationMode, ReplicateHandlingMode, TaskType


def build_df():
    return pd.DataFrame([
        {"sample_id":"g1","group":"g1","target":"10.0","1100":"1.0","1102":"1.1"},
        {"sample_id":"g1","group":"g1","target":"10.0","1100":"1.2","1102":"1.0"},
        {"sample_id":"g2","group":"g2","target":"20.0","1100":"2.0","1102":"2.1"},
        {"sample_id":"g2","group":"g2","target":"20.0","1100":"bad","1102":"2.0"},
        {"sample_id":"g2","group":"g2","target":"20.0","1100":"2.1","1102":"2.2"},
        {"sample_id":"g3","group":"g3","target":"30.0","1100":"3.0","1102":"3.1"},
        {"sample_id":"g3","group":"g3","target":"","1100":"3.1","1102":"3.0"},
    ])


def run(mode: ReplicateHandlingMode):
    backend = Spec4MLPyBackend()
    ds = DatasetService(backend)
    ev = EvaluationService(backend)
    df = build_df()
    sel = DatasetSelection(
        sample_id_column="sample_id",
        target_column="target",
        grouping_column="group",
        spectral_start_index=3,
        task_override=TaskType.REGRESSION,
        replicate_mode=mode,
        replicate_grouping_column="group",
    )
    payload = ds.build_payload(df, "synthetic.csv", sel, drop_invalid_spectral_rows=True)
    res = ev.run(EvaluationMode.LOOCV, payload)
    return payload, res


if __name__ == "__main__":
    p_none, r_none = run(ReplicateHandlingMode.NONE)
    print("NONE", len(p_none.dataframe), len(r_none.predictions_used_for_metrics), [a.name for a in r_none.artifacts])
    p_avg, r_avg = run(ReplicateHandlingMode.AVERAGE_SPECTRA_BEFORE_MODELING)
    print("AVG_SPEC", len(p_avg.dataframe), len(r_avg.predictions_used_for_metrics))
    p_post, r_post = run(ReplicateHandlingMode.AVERAGE_PREDICTIONS_AFTER_MODELING)
    print("AVG_PRED", len(p_post.dataframe), len(r_post.predictions_used_for_metrics), r_post.replicate_aggregation_report.n_groups if r_post.replicate_aggregation_report else None)
