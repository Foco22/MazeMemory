# Re-run Easy (seed cambiada a 7) — 4 escenarios × 30 runs = 120 runs

## Motivo

Los mazes 1 y 2 del diseño original compartían la misma seed (42), lo que introducía
correlación estructural entre Easy y Medium. Para que los 3 mazes tengan seeds
independientes (como ya lo tiene Hard con seed=2024), se re-corre únicamente el
Easy con `seed=7`.

**Cambio aplicado en `src/maze/instances.json`:**
```json
"1": {"seed": 7, "rows": 11, "cols": 11, "difficulty": "easy"}
```

Maze 2 (Medium, seed=42) y Maze 3 (Hard, seed=2024) **no cambian** — sus batch_ids
originales siguen siendo válidos.

---

## 0. Antes de empezar

Confirma que `.env` tiene rellenos:

```
DEEPSEEK_API_KEY=...
SUPABASE_URL=...
SUPABASE_KEY=...
```

Activa el venv:

```bash
cd /home/lenovo/Documentos/PARA/Proyectos/MagisterBirbeck/MazeMemory/MazeMemory
source .venv/bin/activate
```

---

## 1. Los 4 comandos (uno por escenario, difficulty=easy, seed=7)

```bash
python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios baseline --difficulty easy --n-runs 30 --live
ID: cd2a33b8-9992-49ba-9b82-c56c58a60edf

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios shared_memory --difficulty easy --n-runs 30 --live
ID: 9592be0d-2c97-4536-8130-7b50a195da77

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios observer --difficulty easy --n-runs 30 --live
ID:  ffda230f-e5b3-45e9-bc27-2076490212e6

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios shared_memory_observer --difficulty easy --n-runs 30 --live
ID: b49816ab-948a-4e4e-a5f4-39b25ee50eff
```

Corre cada comando uno a la vez. Al terminar cada uno, el sistema imprime el `batch_id`
— pégalo en el espacio correspondiente arriba y en `NOTEBOOK_PLAN.md`.

---

## 2. Batch IDs a reemplazar en el notebook

En `Report/statistic_graph/NOTEBOOK_PLAN.md` y en `Cell 0` del notebook
`report_figures.ipynb`, reemplaza los batch_ids del Easy original por los nuevos:

| Escenario              | Easy (seed=42, VIEJO — descartar)        | Easy (seed=7, NUEVO)     |
|------------------------|------------------------------------------|--------------------------|
| baseline               | `8e776474-5032-4731-ad4a-4b6a944ea41f`   | `<nuevo batch_id>`       |
| shared_memory          | `c12a48b0-da80-4fa0-926e-d71d1bc0ffb0`   | `<nuevo batch_id>`       |
| observer               | `d7efa59a-edfd-4ab3-8cf2-01634202a757`   | `<nuevo batch_id>`       |
| shared_memory_observer | `ed8e4e86-42e8-4229-959e-4d12684bf17d`   | `<nuevo batch_id>`       |

---

## 3. Verificar

```bash
ls results/experiments/*.json | wc -l   # debería haber 120 nuevos archivos
```

Y en Supabase, tabla `experiments`, filtra por los nuevos batch_ids — deben tener
30 filas cada uno con `difficulty = easy`.
