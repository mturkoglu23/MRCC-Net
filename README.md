# MRCC-Net: Leakage-aware evaluation of video-based recognition of stereotypical behaviors in ASD

Code, splits, audit files, and out-of-fold predictions for the manuscript

> *A Leakage-Aware Evaluation of Video-Based Recognition of Stereotypical Behaviors in Autism Spectrum Disorder Using Multi-Rate, Child-Centric Fusion of Video Foundation Models* (submitted to the *Journal of Imaging Informatics in Medicine*).

MRCC-Net classifies pre-trimmed stereotypical behavior clips (arm flapping, hand action, head banging, and spinning) with two **frozen** video foundation models:

- **VideoMAE-L**, at 4, 8, and 16 fps.
- **V-JEPA 2-L**, at 8 fps.

Both models see a full-frame view and a child-centric view; the child is tracked with YOLO11x-pose. This gives eight streams, whose mean-pooled descriptors are concatenated into one 8192-dimensional vector. The only trained component is a standardized, class-balanced, L2-regularized logistic regression with about 33,000 parameters.

The repository also contains the full evaluation protocol:

- Source-grouped 5-fold cross-validation, built after duplicate and linked-recording audits.
- Nested cross-validation.
- Source-group-level permutation tests with Holm correction.
- Random-split, cross-dataset, and external tests.

## Datasets and video links

**No videos are redistributed.** All videos come from public sources; download them from the original URLs, which are listed in the annotation files.

| Dataset | Original source | Links to the videos used here |
|---|---|---|
| SSBD (Self-Stimulatory Behaviour Dataset) [Rajagopalan et al., ICCVW 2013] | https://rolandgoecke.net/research/datasets/ssbd/ | `annotations/segments_ssbd.csv`, `dataset/ssbd.xlsx` |
| ESBD (Expanded Stereotype Behavior Dataset) [Negin et al., Neurocomputing 2021, https://doi.org/10.1016/j.neucom.2021.03.004] | Released as a list of YouTube links | `annotations/segments_esbd.csv`, `dataset/esbd.xlsx` |
| External test: SSBD+ [Lokegaonkar et al., IEEE Healthcom 2023] | https://github.com/sarl-iiitb/ssbdplus-dataset | `external_datasets/clips_manifest.csv`, `external_datasets/all_urls_status.csv` |
| External test: updated SSBD of Wei et al. [Heliyon 2023] | https://github.com/Samwei1/autism-related-behavior | `external_datasets/clips_manifest.csv`, `external_datasets/all_urls_status.csv` |

The annotation tables give the following for every source video:

- URL.
- Annotated behavior intervals, in seconds or as m.ss.
- Number of segments.
- Number of persons visible.

Clips are named `v_<Class>_<ID>[_<k>]_part<n>.mp4` and are placed in `dataset/esbd/`, `dataset/ssbd/`, and `external_datasets/clips/<source>/`. Some URLs may no longer be accessible; `external_datasets/all_urls_status.csv` records their availability on 1 November 2026.

## Repository layout

```
code/                       all scripts (see "Reproducing the results")
dataset/                    esbd.xlsx, ssbd.xlsx: segment annotation tables (place the clips here)
annotations/                segments_{esbd,ssbd}.csv: the same tables as CSV
external_datasets/          external clip manifest, URL availability, overlaps removed by perceptual hashing
results/
  splits/{esbd,ssbd}.json   clip order, labels, source groups, P1 grouped folds (3 x 5) and P2 random folds
  final/manifest_*.csv      clip -> source ID, YouTube ID, intervals, split group, evaluation unit, P1 folds
  final/pred_*.npz          out-of-fold class probabilities for every model and protocol
  final/*.csv               every table of the paper (significance_cluster.csv = Table 8, significance_nested.csv = Table 9, ...)
  figures/source_audit/pairs.json   linked-recording candidates that were inspected visually
  model_revisions.txt       exact Hugging Face commit hashes and the YOLO11x-pose md5
```

The manifest gives the following counts:

- **ESBD:** 195 clips, 79 source IDs, 78 unique YouTube videos, 76 split groups, and 77 evaluation units.
- **SSBD:** 137 clips, with 54 source IDs, groups, and units.

ESBD has three merged linked recordings: HeadBanging_46/47, ArmFlapping_37/41, and ArmFlapping_59/HandAction_05.

## Setup

Requirements:

- Python 3.10.
- An NVIDIA GPU, needed only for feature extraction and fine-tuning; the experiments used an RTX 4090 (24 GB).

```bash
python -m venv venv
venv/Scripts/activate        # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Features and model weights are cached in a working directory. Set it with `OTIZM_WORK`; the default is `D:/otizm_work`.

```bash
export OTIZM_WORK=/path/to/work   # Windows PowerShell: $env:OTIZM_WORK="D:\otizm_work"
```

## Reproducing the results

The statistics, tables, and nested analyses need only the files in this repository; no videos or GPU are required:

```bash
cd code
python stats_cluster.py     # Table 8: cluster sign-flip permutation, cluster bootstrap, Holm
python nested_stats.py      # Table 9: the same comparisons on the nested outer predictions
python ext_ci.py            # video-clustered CIs for the cross-dataset and external tests
python make_manifest.py     # rebuilds the manifest and checks that no source group or YouTube ID crosses folds
```

The full pipeline additionally needs the videos:

```bash
python extract_features_v2.py   # frozen VideoMAE-L / V-JEPA 2-L features, 8 streams (GPU)
python final_experiments.py     # P1, P2, ablations, comparison models
python nested_cv.py             # candidate-and-C selection procedure (P1-N)
python nested_fixed.py          # fixed configurations, C chosen in the inner loop
python robust10.py              # 10-repetition robustness analysis
python cross_dataset_v2.py      # SSBD->ESBD, ESBD->SSBD, and external tests
python finetune.py              # fine-tuned VideoMAE-B / VideoMAE-L baselines (GPU)
python source_audit.py          # linked-recording screening (renders frame pairs for visual inspection)
python tracking_audit.py        # child-tracking audit
```

## Citation

Citation details will be added after publication.

## Funding

This work was supported by the Scientific and Technological Research Council of Türkiye (TÜBİTAK) under the 2209-A University Students Research Projects Support Program (2024/1 call), Project No. 1919B012417912.
