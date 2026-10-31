# Experiment results


| Snapshot | Report | What changed |
| --- | --- | --- |
| [25 October 2026 — performance test](Perfomance_Test_20261025/) | [Key results and findings](Perfomance_Test_20261025/Performance_Test_Report.md) | Frozen 25 Oct models on the same normalized benchmark; Ensemble reaches 98.36% overall accuracy. Training uses unnormalized audio; the raw PART2 stress test is reported separately. |
| [24 October 2026 — performance test](Perfomance_Test_20261024/) | [Key results and findings](Perfomance_Test_20261024/Performance_Test_Report.md) | Frozen 24 Oct models on 9,000 fixed clips; Ensemble reaches 98.80% overall accuracy. Normalization explains the low raw PART2 score. |
| **[24 October 2026 — latest E1](E1_Feature_Engineering_20261024/)** | [Calibrated SVM probabilities](E1_Feature_Engineering_20261024/E1_svm_calibrated_probability.ipynb) | Expanded study from 30,000 to 40,000-clip. |
| **[24 October 2026 — latest E4](E4_Hybrid_Ensemble_20261024/)** | [Calibrated Whisper + SVM probabilities](E4_Hybrid_Ensemble_20261024/E4_03_whisper_svm_calibrated_probability.ipynb) | Expanded study from 30,000 to 40,000-clip. |
| [23 October 2026 — performance test](Perfomance_Test_20261023/) | [MNSC and Common Voice tests](Perfomance_Test_20261023/Performance_Test_Report.md) | Four release models on 6,000 MNSC and 3,000 Common Voice clips, with fixed clip manifests and reproduction scripts. |
| [23 October 2026 — E1](E1_Feature_Engineering_20261023/) | [Calibrated SVM probabilities](E1_Feature_Engineering_20261023/E1_svm_calibrated_probability.ipynb) | 30,000-clip study; calibrated 65-feature SVM was the ML test release candidate. |
| [23 October 2026 — E4](E4_Hybrid_Ensemble_20261023/) | [Calibrated Whisper + SVM probabilities](E4_Hybrid_Ensemble_20261023/E4_03_whisper_svm_calibrated_probability.ipynb) | Calibrated 65-feature acoustic + Whisper fusion SVM was the Ensemble test release candidate. |
| [19 October 2026](E1_Feature_Engineering_20261019/) | [E1 feature selection and model comparison](E1_Feature_Engineering_20261019/E1_feature_engineering_results_report_20261019.md) | Expanded study with 11,486 clips and an updated feature-count selection rule. Selects a 35-feature SVM from the 91-feature pool; retains the full 91-feature SVM as the accuracy reference. |
| [7 October 2026](E1_Feature_Engineering/) | [E1 feature selection and model comparison](E1_Feature_Engineering/E1_feature_engineering_results_report.md) | Earlier study with 2,000 clips. Selects the SVM using all 39 original features. |
