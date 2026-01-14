# RD_BENCH_3 (reference)

Fixed 3-event benchmark for the Ringdown (CVN-RD) module.

Events:
- rin_S150914_pyring_Kerr_221_domega_dtau_221_0M_ft
- rin_S170104_pyring_Kerr_221_domega_dtau_221_0M_ft
- rin_S170814_pyring_Kerr_221_domega_dtau_221_0M_ft

Reference artifacts (results contract):
- reference/results/results_global.json
- reference/results/results_items.csv

Validation:
python tools/compare_results_contract.py --left benchmarks/RD_BENCH_3/reference --check-only
