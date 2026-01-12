Status: **pilot / evolving** (stable locally, not frozen).


# Ringdown-engine (CVN/RD) — минимальный пакет “из коробки”

Это минимальная сборка модуля **Ringdown/CVN** из UCM Calibration Hub: один скрипт, демо-данные и конфиг.
Пакет предназначен для того, чтобы вы могли **распаковать ZIP и сразу запустить** пилотный прогон.

## Установка

Если хотите графики, установите зависимости:

pip install -r requirements.txt


## Быстрый старт (Windows PowerShell)

Из корня:

python -X utf8 modules/ringdown/engine/core/pilot_cvn_rd.py --bench RD_DEMO_221 --tag DEMO --score model_nll --B 200 --root modules/ringdown


Результаты появятся в:
- `.\RUNS\RD_DEMO_221\<timestamp>_RD_DEMO_221_DEMO\results_global.json`
- `.\RUNS\RD_DEMO_221\<timestamp>_RD_DEMO_221_DEMO\results_event.csv`
Also emits results/ artifacts compatible with tools/results_contract.md.

## Структура

- `engines/pilot_cvn_rd.py` — движок (CLI)

## Замечание про демо-данные

Файлы `posteriors_csv/*.csv` — синтетические, сделаны только для проверки работоспособности конвейера.
Для реального анализа замените их на ваши CSV/выгрузки из H5 (с колонками `delta_f`, `delta_tau`, и по возможности `mf_solar`, `final_spin`).

## Лицензия

