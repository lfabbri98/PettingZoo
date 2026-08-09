# Tennis RL Simulator

Simulatore headless di tennis per reinforcement learning, con un agente PPO
che controlla il giocatore basso contro l'avversario euristico classico.

## Avvio

Installa le dipendenze di training e avvia un run base:

```bash
uv sync --extra train
uv run --extra train python train.py --episodes 200 --checkpoint checkpoints/tennis_ppo.pt
```

Gli artefatti in `checkpoints/` sono ignorati da Git. Per una prova rapida:

```bash
uv run --extra train python train.py --episodes 2 --rollout-episodes 1
```

Il trainer raccoglie episodi con la policy corrente, calcola GAE e applica
aggiornamenti PPO con clipping, loss del critic, bonus di entropia, minibatch e
gradient clipping. Il checkpoint contiene `model_state_dict` e gli
iperparametri PPO.

Per visualizzare la partita con la policy classica:

```bash
uv run --extra render python main.py
```

## Test

```bash
uv run pytest -q
```
