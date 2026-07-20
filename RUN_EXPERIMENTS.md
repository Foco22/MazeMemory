# Cómo correr el diseño experimental completo

4 escenarios × 3 laberintos × 30 runs = **360 runs totales**, guardados en Supabase.

Modelo: `deepseek/deepseek-v4-flash` (provider `deepseek`).

## 0. Antes de empezar

Confirma que `.env` tiene rellenos (no vacíos):

```
DEEPSEEK_API_KEY=...
SUPABASE_URL=...
SUPABASE_KEY=...
```

Párate en la raíz del proyecto y activa el `.venv` del proyecto (el `python`/`python3`
del sistema apunta a Anaconda, que **no** tiene `litellm` instalado — te va a tirar
`ModuleNotFoundError` si no usas el venv):

```bash
cd /home/lenovo/Documentos/PARA/Proyectos/MagisterBirbeck/MazeMemory/MazeMemory
source .venv/bin/activate
```

Deja esa terminal (o cada terminal nueva que abras para esto) con el venv activado —
vas a ver `(.venv)` al inicio del prompt. Los comandos de abajo asumen que ya lo
activaste.

## 1. Los 12 comandos (uno por escenario × dificultad)

Cada uno corre 30 runs de esa combinación y guarda en Supabase (no lleva `--no-db`).
Corre a Supabase por defecto — no hace falta ninguna flag extra para eso.

```bash
python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios baseline --difficulty easy --n-runs 30 --live
ID: 8e776474-5032-4731-ad4a-4b6a944ea41f

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios baseline --difficulty medium --n-runs 30 --live
ID: 145cfe0d-ee40-4654-a7d9-fbb985a4535e

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios baseline --difficulty hard --n-runs 30 --live
ID: 6abd1a94-f778-4272-9995-39d90f7f2b7e

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios shared_memory --difficulty easy --n-runs 30 --live
ID: c12a48b0-da80-4fa0-926e-d71d1bc0ffb0

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios shared_memory --difficulty medium --n-runs 30 --live
ID: 95fa43ea-ddac-409c-824f-9b28890d0325

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios shared_memory --difficulty hard --n-runs 30 --live
ID: 7036b35f-011e-4350-bf56-97ee6a538a7d

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios observer --difficulty easy --n-runs 30 --live
ID: d7efa59a-edfd-4ab3-8cf2-01634202a757

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios observer --difficulty medium --n-runs 30 --live
ID: 4fd05ceb-37a5-4065-88c5-89a8e30d69d9

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios observer --difficulty hard --n-runs 30 --live
ID; a3ccc94b-d925-4b70-9701-c9037cd4828b

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios shared_memory_observer --difficulty easy --n-runs 30 --live
ID: ed8e4e86-42e8-4229-959e-4d12684bf17d

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios shared_memory_observer --difficulty medium --n-runs 30 --live
ID: c0869925-9952-4685-b965-48dc3ad4f297

python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios shared_memory_observer --difficulty hard --n-runs 30 --live
ID: 7d0dddad-f60b-4d44-acea-0416831ffc6c
```

Con `--live` ves el laberinto y las posiciones de los 3 agentes actualizándose en tiempo
real en la terminal, además del resumen (`RUN SUMMARY`) al terminar cada run. Corre cada
comando uno a la vez en una terminal que estés mirando — deja terminar uno antes de
lanzar el siguiente.

**Por qué 12 comandos y no uno solo:** cada uno guarda su resultado (JSON local + fila en
Supabase) por run, así que si uno falla o lo interrumpes, sabes exactamente cuál
combinación repetir sin volver a correr las demás. (También podrías correr todo en un
solo comando omitiendo `--scenarios` y `--difficulty` — por defecto corre las 4 × 3 — pero
si se interrumpe a la mitad, no hay forma limpia de saber por dónde quedó.)

## 2. Alternativa sin `--live` (solo si en algún momento no quieres verlo)

`--live` usa códigos de control de pantalla pensados para una terminal real — si lo
combinas con `nohup`/redirección a un archivo, el log sale corrupto (con basura de
códigos ANSI en vez de texto legible). Si prefieres correr algo desatendido en segundo
plano, quita `--live` del comando:

```bash
mkdir -p logs

nohup python experiments/run.py --model deepseek/deepseek-v4-flash --provider deepseek --scenarios baseline --difficulty easy --n-runs 30 > logs/baseline_easy.log 2>&1 &
```

Repite cambiando `--scenarios`/`--difficulty` y el nombre del log para cada una de las 12
combinaciones. Para ver el progreso de un log:

```bash
tail -f logs/baseline_easy.log
```

## 3. Verificar que todo se guardó

```bash
ls results/experiments/*.json | wc -l   # debería llegar a 360 al terminar todo
```

Y en Supabase, revisa la tabla `experiments` — debería tener 360 filas nuevas
(`scenario`, `maze_id`, `run_number` por fila).
