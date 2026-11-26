# Current Dataset Status

Updated: 2026-11-25

## A/B/C/D datasets

- A: CWRU 12 kHz drive-end, 0 HP.
- B: JNU Bearing Dataset, 600 rpm; selected normal, inner-race, and outer-race
  records are under `external/jnu/raw/`.
- C: UESTC Bearing Dataset, 800 rpm; selected normal, inner-race, and outer-race
  records are under `external/uestc/raw/`.
- D: Ottawa Bearing Dataset v2; five health conditions × four speed profiles ×
  three trials = 60 records, sampled at 200 kHz for 10 seconds each.
- Ottawa raw archive: `ottawa/raw/ottawa_bearing_v2.zip` (799,407,431 bytes).
- Archive SHA-256:
  `4e5538e71692c83e9f5a4338808983f2a9bb00bfdc4a4e4a9627b616f5794eb1`.
- Extracted records: `ottawa/raw/extracted/` (60 MAT files).
- Ottawa source DOI: `10.17632/v43hmbwxpm.2`.
- For the shared three-class experiment, use healthy (`H-*`), inner-race
  (`I-*`), and outer-race (`O-*`) records across profiles A–D and trials 1–3;
  exclude ball-fault (`B-*`) and combined-fault (`C-*`) records.
- Canonical manifest and processing configuration are in `abc/manifest.csv` and
  `abc/dataset_config.json`.

The completed run used A=CWRU → B=JNU → C=UESTC → D=Ottawa →
A_return=CWRU.
