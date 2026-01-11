# Reproducibility Bundle Template

Use this template when publishing an external reproducibility archive (e.g., Zenodo)
for any UCM-T module.

---

## Bundle title

**(Example)** UCM-T — <ModuleName> Reproducibility Bundle (vX.Y)

## Abstract (short)

1–2 paragraphs describing:
- the scientific purpose of the bundle,
- what is included,
- what result it reproduces.

## Contents

List the top-level files/folders in the archive, for example:

- `engine/` — code used for the run
- `data/` — input data (or scripts to fetch it)
- `configs/` — parameter sets / YAML / JSON
- `runs/` — outputs (summaries; large artifacts compressed)
- `scripts/` — launch scripts (PowerShell/Bash) and helpers
- `README.md` — exact run instructions
- `requirements.txt` or `environment.yml`

## Reproduce

### System requirements
- OS:
- Python version:
- CPU/RAM:
- Optional: GPU:

### Installation
Provide exact commands.

### Run
Provide exact commands to reproduce key outputs.

### Expected outputs
List the expected files and key numbers/figures to compare.

## Versioning and provenance

- Repository: `Vladimir-UCMT/UCM-T`
- Module: `<modules/...>`
- Code revision (commit hash):
- External archive: (Zenodo DOI or GitHub Release tag)
- Date:

## Notes / limitations

Document known limitations, failure modes, and numerical stability notes.

## Citation

Provide the preferred citation text and persistent identifiers (DOI).
