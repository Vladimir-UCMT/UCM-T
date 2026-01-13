# Tools

## Results contract comparator

UCM-T modules emit a lightweight **results contract**: a run-level JSON summary and
an item-level CSV table. The comparator validates these artifacts and can compare
runs side-by-side.

- Contract definition: [`results_contract.md`](results_contract.md)
- Comparator script: [`compare_results_contract.py`](compare_results_contract.py)

### Expected layout

A run directory should include:

```
results/
  results_global.json
  results_items.csv
```

If the `results/` folder is missing, the comparator falls back to legacy root-level
files (`results_global.json`, `results_items.csv`) and reports a warning.
The comparator accepts `results_global.json` with or without a UTF-8 BOM.

### Validate a single run

```
python tools/compare_results_contract.py --left <RUN_DIR> --check-only
```

### Compare two runs

```
python tools/compare_results_contract.py --left <RUN_DIR_A> --right <RUN_DIR_B>
```

### Write a report

```
python tools/compare_results_contract.py \
  --left <RUN_DIR_A> \
  --right <RUN_DIR_B> \
  --out comparison_report.md
```

If the output path ends with `.json`, the report is written as JSON instead of
Markdown.

### Typical workflows

- Validate a single run before archiving artifacts.
- Compare two different engines (e.g., Rotation Curves vs Ringdown) over the same
  dataset.
- Diff regression runs to identify shifts in per-item metrics.
