# Tennis simulator

Primo passo: un campo 2D e due giocatori che possono muoversi soltanto lungo la
linea di fondo. L'ambiente è separato dalla demo e non legge input da tastiera.
I valori configurabili sono in `parameters.yml`.

## Avvio

```bash
uv sync
uv run python main.py
```

Per ora `main.py` usa due semplici politiche automatiche solo per mostrare il
movimento. Gli agenti controlleranno invece l'ambiente con:

```python
environment.step(top_action, bottom_action, delta_time)
```

Ogni azione è un intero: `-1` (sinistra), `0` (fermo), `1` (destra).
