# Сохранённые модели

Карим сохраняет checkpoint после обучения. Аслан фиксирует в результатах имя checkpoint, который использовался при оценке и записи демо.

## Что лежит в папке

| Файл | Что это |
| --- | --- |
| `qmix_spread_n2_seed0.pt` | **Основная модель** (seed 0): веса сети агента и смешивающей сети, параметры запуска, версии библиотек |
| `qmix_spread_n2_seed0.json` | То же описание в читаемом виде и SHA-256 файла `.pt` |
| `qmix_spread_n2_seed1.pt`, `..._seed2.pt` и их `.json` | Те же настройки, seed 1 и 2. Эти seed (как и 0) участвовали в подборе настроек |
| `qmix_spread_n2_seed4.pt`, `..._seed5.pt` и их `.json` | Те же настройки, seed 4 и 5, которые в подборе настроек не участвовали: честная проверка, что обучение повторяется |

Имя файла: `qmix_spread_n<число агентов>_seed<seed обучения>`. Каждая модель обучалась 200 000 шагов среды (8 000 эпизодов). Числа по каждой — в `results/qmix_training_report.md` и `results/tables/qmix_training_check_heldout.csv`. Если сравниваете несколько запусков, берите все пять моделей, а не только лучшую.
В отчёте о результатах указывайте имя файла и первые 12 символов SHA-256 из JSON.

## Как запустить модель (для Аслана)

Из корня репозитория (в Colab — после `git clone` и `pip install -r requirements.txt`):

```python
import sys
sys.path.insert(0, "src")

import numpy as np
from qmix_spread.checkpoint import load_checkpoint
from qmix_spread.env import SpreadEnv
from qmix_spread.rollout import QMIXPolicy, RandomPolicy, evaluate, run_episode

CHECKPOINT = "models/qmix_spread_n2_seed0.pt"
model = load_checkpoint(CHECKPOINT)  # only loads weights, no training
env = SpreadEnv(model.config)  # same environment settings as in training
qmix = QMIXPolicy(model.agent_net)  # execution needs only the agent network
random_agents = RandomPolicy(env.n_agents, env.n_actions)

# 50 fixed start positions that were not used in training or in the training-time evaluation.
seeds = list(range(20_000, 20_050))
qmix_returns = evaluate(env, qmix, seeds)  # epsilon = 0 by default: no random actions
random_returns = evaluate(env, random_agents, seeds, rng=np.random.default_rng(0))
print(f"QMIX:   {qmix_returns.mean():.2f} +- {qmix_returns.std():.2f}")
print(f"random: {random_returns.mean():.2f} +- {random_returns.std():.2f}")

# One episode as images for the video: 26 frames of 700 x 700 pixels.
video_env = SpreadEnv(model.config, render_mode="rgb_array")
episode = run_episode(video_env, qmix, seed=20_000, record_frames=True)
frames = episode.frames
print(len(frames), frames[0].shape, f"episode return {episode.total_reward:.2f}")
```

Что важно знать:

- `evaluate` возвращает награду команды за каждый эпизод: сумму 25 наград за шаги. Награда отрицательная, ближе к нулю — лучше.
- Один и тот же seed даёт одну и ту же начальную расстановку агентов и точек, поэтому QMIX и случайные агенты сравниваются на одинаковых начальных состояниях.
- Какие числа уже заняты. Seed обучения: 0–5 (3 — только в пробных запусках). Начальные расстановки (seed среды): оценка во время обучения — 10 000–10 031; итоговая проверка Карима — 30 000–30 099; пример тензоров в notebook — 40 000–40 003. Для своей оценки берите другие расстановки, например 20 000–20 049, как в примере.
- Обучить модель с новым seed: `PYTHONPATH=src python -m qmix_spread.train --seed 6` (около 5 минут на одном ядре процессора). Файлы появятся в `models/`, `results/` и `configs/` с номером seed в имени. Запуск с seed 0–5 перезапишет уже сохранённые файлы этого seed (веса получатся те же, но изменятся дата и контрольная сумма).
