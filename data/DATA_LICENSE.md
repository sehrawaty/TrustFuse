# Data provenance and terms

`data/processed/` holds **processed wrist-sensor streams** derived from two public research
datasets. The original archives are not included: WESAD is 2.25 GB and PPG-DaLiA is 2.86 GB,
both far above GitHub's file-size limit. Each subject file keeps the Empatica E4 wrist streams
at their native sampling rates: BVP at 64 Hz, ACC at 32 Hz, and EDA and skin temperature at
4 Hz. The labels are downsampled to 4 Hz. `manifest.json` in each folder lists the SHA-256
checksum of every file. `python -m trustfuse verify` checks them.

To regenerate the files from the original archives, put `WESAD.zip` and/or PPG-DaLiA's inner
`data.zip` into `data/raw/` and run `python -m trustfuse prepare`. The setup script can also
do this with `-FromRaw`.

## WESAD — Wearable Stress and Affect Detection

* Source: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download (UCI id 465, doi:10.24432/C57K5T)
* Citation: P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, K. Van Laerhoven,
  "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection",
  *Proc. ICMI 2018*, pp. 400–408. doi:10.1145/3242969.3242985
* Terms: the dataset is provided by its authors for academic, non-commercial research. Please
  follow the acknowledgment policy in the original WESAD readme. The processed files are
  shared here only for reproducing this academic project.

## PPG-DaLiA

* Source: https://archive.ics.uci.edu/dataset/495/ppg+dalia
* Citation: A. Reiss, I. Indlekofer, P. Schmidt, K. Van Laerhoven, "Deep PPG: Large-Scale
  Heart Rate Estimation with Convolutional Neural Networks", *Sensors* 19(14):3079, 2019.
  doi:10.3390/s19143079
* License: Creative Commons Attribution 4.0 International (CC BY 4.0).

## What was changed

The changes are format changes only; no sample values were altered:

* Only the wrist (E4) streams were kept.
* Accelerometer values are stored as int8 counts of 1/64 g, which is the sensor's native
  resolution, so this is lossless.
* WESAD's 700 Hz protocol labels were downsampled to 4 Hz by taking the label at the centre
  of each 0.25 s bin.
* Streams were cropped to their common duration.
