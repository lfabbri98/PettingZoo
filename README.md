# Tennis RL Simulator

Simulatore headless di tennis per reinforcement learning. Il wrapper
`TennisEnv` controlla il giocatore basso contro un avversario euristico
classico iniettabile.

## Avvio

Il progetto include il simulatore e il wrapper `TennisEnv`, pronti per essere
collegati a un nuovo algoritmo di reinforcement learning.

Per visualizzare una partita con la policy classica:

```bash
uv run --extra render python main.py
```

## Test

```bash
uv run pytest -q
```
