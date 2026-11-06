# MRCC-Net: Leakage-aware evaluation of video-based recognition of stereotypical behaviors in ASD

This repository contains the code, splits, audit files, and out-of-fold predictions for the manuscript

> *Leakage-Aware Recognition of Stereotypical Behaviors in Autism Using Multi-Rate Child-Centric Fusion of Video Foundation Models* (submitted to the *Journal of Imaging Informatics in Medicine*).

## Method in brief

MRCC-Net classifies pre-trimmed behavior clips with two **frozen** video foundation models:

- **VideoMAE-L** (Kinetics-400), at 4, 8, and 16 fps.
- **V-JEPA 2-L** (SSv2), at 8 fps.

Each model sees two views of every clip: the full frame and a child-centric crop around the child, who is tracked with YOLO11x-pose. Each view is also horizontally flipped. This gives eight streams.

For each stream, the window embeddings are averaged over windows and flips into one 1024-d descriptor. The eight descriptors are concatenated into an 8192-d clip descriptor.

The only trained component is a standardized, class-balanced, L2-regularized logistic regression (C = 0.05), with 32,772 parameters for four classes. A video-level decision averages the clip probabilities.

The classes are ArmFlapping, HeadBanging, and Spinning on both datasets, plus HandAction on ESBD.

## Evaluation protocol

| Protocol | Definition |
|---|---|
| P1 (primary) | Source-grouped stratified 5-fold cross-validation, 3 repeats (seeds 0, 1, 2), after merging duplicate and linked recordings. A 10-repeat version is run for the frozen models. |
| P2 | Random clip-level stratified 5-fold cross-validation, 3 repeats. Used for comparison with the literature. |
| P3 | Cross-dataset tests (SSBD→ESBD and ESBD→SSBD) on the three shared classes. |
| P4 | External test: trained on SSBD + ESBD, tested on 49 external clips (30 videos). |
| P1-N | Nested source-grouped cross-validation with a 4-fold inner loop. It has two forms: (a) fixed configurations with only C chosen in the inner loop; (b) joint selection over 10 candidate configurations × 5 C values. |

Significance is tested with a paired sign-flip permutation test that treats each source group as one unit. Confidence intervals come from a cluster bootstrap. Holm correction is applied to the eight primary comparisons: MRCC-Net vs. four single-backbone probes, on two datasets.

The MRCC-Net configuration and C were chosen during the design phase on P1. The P1 p-values are therefore supportive rather than confirmatory, as discussed in the paper.

## Datasets and video links

**No videos are redistributed in this repository.**

| Dataset | Source | Links to the videos used here |
|---|---|---|
| SSBD, Self-Stimulatory Behaviour Dataset (Rajagopalan et al., ICCVW 2013) | Official page: https://rolandgoecke.net/research/datasets/ssbd/. This study used the copy of the dataset distributed on Kaggle. | `annotations/segments_ssbd.csv`, `dataset/ssbd.xlsx` |
| ESBD, Expanded Stereotype Behavior Dataset (Negin et al., Neurocomputing 2021, https://doi.org/10.1016/j.neucom.2021.03.004) | Released as a list of YouTube links. | `annotations/segments_esbd.csv`, `dataset/esbd.xlsx` |
| External test: SSBD+ (Lokegaonkar et al., IEEE Healthcom 2023, https://doi.org/10.1109/HEALTHCOM56612.2023.10472349) | https://github.com/sarl-iiitb/ssbdplus-dataset | `external_datasets/clips_manifest.csv` |
| External test: updated SSBD list of Wei et al. (Heliyon 2023, https://doi.org/10.1016/j.heliyon.2023.e16763) | https://github.com/Samwei1/autism-related-behavior | `external_datasets/clips_manifest.csv` |

The external set also contains 3 clips from SSBD videos that are missing from the local copy.

The annotation tables (`annotations/segments_*.csv`, and the same sheets in `dataset/*.xlsx`) give, for every source video used:

- the URL;
- the annotated behavior intervals, written either in seconds or as m.ss;
- the number of segments and their lengths;
- the number of persons visible;
- a short subtype note, in Turkish.

`external_datasets/clips_manifest.csv` gives the source video and the start and end times of every external clip. `external_datasets/all_urls_status.csv` records whether the SSBD (75), SSBD+ (36), and Wei et al. (61) URLs were still accessible on 1 November 2026. `external_datasets/clips_overlap_with_local/OVERLAP.csv` lists the external clips that were removed because they share a source video with the local data.

The code expects the clips at the following locations:

| Dataset | Folder | Example file name |
|---|---|---|
| ESBD | `dataset/esbd/` | `v_HeadBanging_46_part2_slice1.mp4` |
| SSBD | `dataset/ssbd/` | `v_<Class>_<ID>_part<n>.mp4` |
| External test | `external_datasets/clips/{ssbd,ssbdplus,wei2023}/` | `v_ArmFlapping_P16_part1.mp4` |

All clip names follow `v_<Class>_<ID>[_<k>]_part<n>[_slice<m>].mp4`. The source ID is `<Class>_<ID>`.

## Repository layout

```
code/                        all scripts (see below)
dataset/{esbd,ssbd}.xlsx     segment annotation sheets (place the clips in dataset/esbd/ and dataset/ssbd/)
annotations/                 the same annotation sheets as CSV
external_datasets/           external clip manifest, URL availability, overlaps removed by perceptual hashing
results/
  splits/{esbd,ssbd}.json    clip order, labels, source groups; P1 grouped folds and P2 random folds (3 repeats x 5 folds)
  final/                     tables, manifests, and out-of-fold predictions used in the paper (see below)
  figures/source_audit/pairs.json   the 36 (ESBD) + 30 (SSBD) cross-group pairs inspected for linked recordings
  model_revisions.txt        exact Hugging Face commit hashes and the md5 of yolo11x-pose.pt
```

The key files in `results/final/` map to the paper as follows:

| File | Content |
|---|---|
| `manifest_{esbd,ssbd}.csv`, `manifest_counts.csv` | For each clip: source ID, YouTube ID, annotated intervals, split group, evaluation unit (group × class), and P1 fold in each repeat. ESBD has 195 clips, 79 source IDs, 78 unique videos, 76 split groups, and 77 evaluation units. SSBD has 137 clips and 54 source IDs, groups, and units. |
| `main_results.csv`, `baselines_v1.csv`, `finetune.csv` | Tables 3–4: frozen single-backbone probes and MRCC-Net; DINOv2-B, X-CLIP-B/16, pose velocity spectrum, and RePAF-Net; fine-tuned VideoMAE-B/L. |
| `leakage.csv` | Table 6. |
| `ablation.csv` | Table 7. |
| `significance_cluster.csv` | Table 8, plus exploratory ablation and fine-tuning tests. |
| `significance_nested.csv` | Table 9. |
| `robust10.csv`, `robust10_significance.csv` | 10-repeat analysis. |
| `sensitivity_C.csv`, `sensitivity_pooling.csv` | Tables 10–11. |
| `nested_cv.csv`, `nested_cv_selection.csv`, `nested_fixed.csv`, `nested_fixed_C.csv` | P1-N results. |
| `cross_dataset.csv`, `external_ci.csv`, `external_nn_check.csv` | Table 12 and its confidence intervals; nearest-neighbor check of the external clips. |
| `cost.csv` | Table 13. |
| `per_class.csv`, `confusion_*.csv` | Table 14 and Fig. 5. |
| `tracking_audit.csv`, `tracking_audit_manual.csv` | Child-tracking audit. |
| `pred_{esbd,ssbd}_{grouped,random}_<model>.npz` | Out-of-fold probabilities (`p`: repeats × clips × classes, `y`, `groups`) for the frozen probes, MRCC-Net, and ablations A1–A8. |
| `pred_ft_*.npz` | Fine-tuned VideoMAE-B/L. |
| `pred_cross_*.npz` | P3/P4. |
| `pred_nested_*.npz` | P1-N, with C chosen in the inner loop (`P`, `y`, `g`). |
| `significance.csv` | Legacy McNemar tests written by `final_experiments.py`. They are **not** used in the paper. |

## Setup

Requirements:

- Python 3.10. Package versions are in `requirements.txt`.
- For `torch==2.11.0`, choose the build that matches your CUDA version.
- A GPU is needed only for feature extraction, the cost measurement, and fine-tuning. The experiments used one NVIDIA RTX 4090 (24 GB).

```bash
python -m venv venv
venv/Scripts/activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Model weights and features:

- **Hugging Face weights** are downloaded into the standard Hugging Face cache. The exact revisions are listed in `results/model_revisions.txt`.
- **YOLO11x-pose:** download the Ultralytics weights and place them at `$OTIZM_WORK/weights/yolo11x-pose.pt`.
- **Features** are cached in `$OTIZM_WORK/features`.

`OTIZM_WORK` defaults to `D:/otizm_work`. To change it:

```bash
export OTIZM_WORK=/path/to/work      # Windows PowerShell: $env:OTIZM_WORK="D:\otizm_work"
```

## Reproducing the results

The statistics can be recomputed from the files in this repository alone; no videos or GPU are needed. Run these from `code/`:

```bash
python stats_cluster.py    # Table 8: source-group sign-flip permutation, cluster bootstrap, Holm
python nested_stats.py     # Table 9: the same comparisons on the nested outer predictions
python ext_ci.py           # video-clustered CIs for the cross-dataset and external tests
```

These three commands reproduce `significance_cluster.csv`, `significance_nested.csv`, and `external_ci.csv` exactly.

The full pipeline needs the clips. Run the steps in this order, from `code/`:

```bash
python make_manifest.py                   # manifest; checks that each source group stays in one fold and no YouTube ID spans two groups
python extract_features.py                # stage-1 features: VideoMAE-L, X-CLIP, DINOv2, pose (v1 baselines and RePAF-Net)
python extract_features_v2.py             # stage-2 features: VideoMAE-L/H and V-JEPA 2-L/g, full + child views, 4/8/16 fps
python train_eval.py --dataset esbd --protocol grouped --methods lr_xclip lr_dino lr_pose_spec repaf
                                          # repeat for --dataset ssbd and --protocol random
python final_experiments.py               # P1/P2 for the frozen probes, MRCC-Net, and the ablations
python sensitivity.py                     # sensitivity to C and to window pooling
python robust10.py                        # 10-repeat analysis
python nested_cv.py                       # P1-N, candidate-and-C selection
python nested_fixed.py                    # P1-N, fixed configurations
python cross_dataset_v2.py                # P3 and P4
python finetune.py --model videomae-B     # fine-tuned VideoMAE-B, P1 and P2
python finetune.py --model videomae-L --protocols grouped
python stats_cluster.py && python nested_stats.py && python ext_ci.py
python cost.py                            # Table 13 (GPU timing)
python tracking_audit.py                  # child-tracking audit (contact sheets for visual inspection)
python source_audit.py                    # linked-recording screen (frame pairs for visual inspection)
python make_figures.py                    # Figs. 2-5
python make_architecture_v3.py            # Fig. 1
```

Notes on the scripts:

- **Bundled splits:** the folds in `results/splits/` are the ones used in the paper. `make_splits.py` regenerates them from the clips and the linked-recording merges in `common.GROUP_ALIAS`.
- **Appended results:** `train_eval.py` appends its rows to `results/cv_results.csv`. The paper reports the last three repeats of each setting, which are the rows in `results/final/baselines_v1.csv`.
- **Development scripts:** `cross_dataset.py`, `eval_compact.py`, and `explore_fusion.py` are kept for transparency but are not needed to reproduce the paper.
- **Online Resource 1:** `make_supplement.py` builds the reviewer package from a local working copy.

## Citation

Citation details will be added after publication.

## Funding

This work was supported by the Scientific and Technological Research Council of Türkiye (TÜBİTAK) under the 2209-A University Students Research Projects Support Program (2024/1 call), Project No. 1919B012417912.
