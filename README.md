# Tennis RL Simulator

Simulatore headless di tennis pensato per esperimenti di reinforcement
learning. Il core fisico e il wrapper non dipendono da Pygame; Pygame è usato
solo dall'interfaccia grafica.

## Installazione

Per core e test:

```bash
uv sync --group dev
uv run pytest
```

Per avviare anche la simulazione grafica:

```bash
uv sync --extra render --group dev
uv run python main.py
```

## API headless

```python
from tennis_env import TennisEnv

env = TennisEnv()
observation, info = env.reset(seed=42)

while True:
    action = (0.0, -0.5, 0.8, 0.0)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

Un episodio è una partita vinta dal primo giocatore che raggiunge 11 punti.
Il giocatore basso è controllato dall'agente; quello alto usa per default la
`classic_policy`. Una policy diversa può essere passata con
`opponent_policy=`.

### Azione

L'azione contiene quattro valori:

1. `move_x` in `[-1, 1]`;
2. `move_y` in `[-1, 1]`;
3. velocità d'uscita desiderata in `[0, 1]`, convertita nell'intervallo
   `min_shot_speed`–`max_speed`;
4. angolo in `[-1, 1]`, convertito nell'intervallo consentito al giocatore.

Per accelerare il training, ogni azione viene mantenuta per `frame_skip=4`
tick fisici. Il valore è configurabile; la demo grafica usa `frame_skip=1`.

### Osservazione

L'osservazione è una tupla di 15 valori:

- posizione e velocità dell'agente: 4 valori;
- posizione e velocità dell'avversario: 4 valori;
- posizione e velocità della pallina: 4 valori;
- punteggio normalizzato di agente e avversario: 2 valori;
- turno attivo: `1` agente, `-1` avversario.

Le coordinate sono normalizzate rispetto al campo e lo stato è sempre
presentato dal punto di vista del giocatore controllato.

### Ricompensa e fine episodio

- `+1` quando l'agente segna un punto;
- `-1` quando segna l'avversario;
- `terminated=True` quando un giocatore raggiunge 11;
- `truncated=True` quando viene raggiunto il limite di decisioni.

`info` contiene punteggio, autore dell'ultimo punto, vincitore, numero di
decisioni e numero di tick fisici.
