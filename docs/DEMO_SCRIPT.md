# Demo video script (target 2:30)

| Time | On screen | Say |
|---|---|---|
| 0:00 | Title slide | "I am Diar. This is a smarter traffic light that combines three AI topics." |
| 0:15 | Problem slide | "Fixed 30/30 signals waste green time when traffic is light and build queues when it is heavy. The sensor is noisy, so I estimate the traffic state." |
| 0:35 | Terminal: `python src/traffic_controller.py` | "Step 1, Bayes: a High reading raises P(Heavy) from 30% to 64.6%." |
| 0:55 | Same output, hill climbing block | "Step 2, hill climbing: it walks S1, S2, S3 and picks 40 s Main / 20 s Side, 36.46 s expected delay." |
| 1:15 | Same output, MDP block | "Step 3, MDP: the belief-weighted Bellman value picks Extend Green." |
| 1:30 | Browser: `web/index.html`, click High, then Reset and Low | "Same model in the browser. With a Low reading it keeps 30/30. The lights run the chosen timing." |
| 2:00 | Move the prior slider | "A different prior changes the answer, so the algorithms really depend on the belief." |
| 2:10 | Terminal: `python -m unittest discover -s tests` | "13 tests pass, including hill climbing against brute force." |
| 2:25 | Last slide | "All numbers are assumed. Next: real traffic counts and an HMM. Thank you." |
