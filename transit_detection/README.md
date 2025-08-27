# High-Level Dataset Build & Processing Workflow

## 1. Build Dataset

1. **Set conditions** in `config_build_dataset.yaml`.
2. **Run**:

   ```bash
   python build_dataset.py
   ```

### Notes
* `build_dataset.py relies on various masking procedures, each with tuneable params, to compute examples with minimal noise, including a negative disposition transit mask, weak secondary transit mask, 
    * Due to dataset construction changes, not all plotting functionality is functional - often best done after construction for visualization - or construction on a subset for testing w/ post processing.
    * Due to dataset construction changes, logging was heuristic and improvements should be made if neccesary.
* `build_dataset_t_sr.py` contains the **previous** dataset build logic — replaced by **target-level masking** to prevent erroneous example overlap and label noise.
* Avoid **too many targets per shard** — may cause **core dumps** or interrupted builds.
* Output structure (current config example):

  ```
  dataset_name/
      tfrecords/
          data_tbl_0001-XXXX.csv
          ...
          data_tbl_XXXX-XXXX.csv
          raw_shard_0001-XXXX.tfrecord
          ...
          raw_shard_XXXX-XXXX.tfrecord
  ```

> **Important**: For all subsequent steps, `num_shards` in relevant scripts must match `XXXX` from the original dataset.
> If using multiprocessing: `num_processes` ≤ number of available CPU cores.

---

## 2. Split Dataset

Run:

```bash
python dataset_handling/split_tfrec_dataset.py
```

Output:

```
dataset_name_split/
    tfrecords/
        train/
            train_shard_0001-XXXX.tfrecord
        val/
        test/
```

---

## 3. Compute Normalization Statistics

Run:

```bash
python norm_pipeline/compute_train_stats.py
```

### Notes

* Computed at **target level**.
* Max examples per TCE for stats: **4**.
* Produces `.npy` file with training stats for normalization.

---

## 4. Normalize Dataset

Run:

```bash
python norm_pipeline/norm_tfrec_dataset_split.py
```

Shards transform:

```
raw_train_shard_0001-XXXX.tfrecord
    ↓
norm_train_shard_0001-XXXX.tfrecord
```

> **Pipeline dependency**: Naming format must stay the same unless the pipeline is updated.

---

## 5. Train Model

Run:

```bash
python keras_model/train_model.py
```

* Uses config in `keras_model/config_train.yaml`.

---

## 6. Optional: Filter Dataset

Run:

```bash
python dataset_handling/remove_examples_by_condition.py
```

* Filtering logic is **manually defined** in `process_shard`.

---
