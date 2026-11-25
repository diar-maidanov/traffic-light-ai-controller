# Mini Traffic Light Controller (AI Fundamentals Final Project, topic 08)

**Author:** Diar

## Problem
A single intersection (Main road and Side road) sees changing traffic. A fixed 30/30 s signal wastes green time when traffic is light and builds long queues when Main is heavy. The controller reads one noisy vehicle sensor, estimates how heavy traffic really is, picks a signal timing, and decides whether to extend the current green.

## How the three algorithms work together
| Step | Algorithm (course topic) | Input | Output |
|---|---|---|---|
| 1 | **Bayesian inference** (probabilistic reasoning) | prior P(Heavy), sensor reading | P(Heavy traffic) |
| 2 | **Hill climbing** (local search) | 4 timing settings, delay table, P(Heavy) | selected timing |
| 3 | **MDP / Bellman** (sequential decisions) | rewards, transitions, P(Heavy) | Extend Green or Switch |

The Bayesian belief is the shared input of steps 2 and 3. Without it, the prior alone would pick timing S2 and the action Switch; with a "High" reading the controller picks S3 and Extend Green. The algorithms are therefore not independent demos.

## Run
```
python src/traffic_controller.py                 # one "High" reading (default)
python src/traffic_controller.py --obs Low       # one "Low" reading
python src/traffic_controller.py --obs High High # two readings in a row
python src/traffic_controller.py --prior 0.5     # change the prior
```
Needs only Python 3 (standard library). To see the interactive website, open `web/index.html` in any browser (no server or internet needed).

## Assumptions (hypothetical, no dataset)
- Prior P(Heavy) = 0.30. Sensor: P(High|Heavy) = 0.85, P(High|Light) = 0.20.
- 4 settings (Main/Side green): S1 = 20/40, S2 = 30/30, S3 = 40/20, S4 = 50/10 seconds.
- Average delay (s/vehicle), Heavy: 70, 52, 40, 46. Light: 30, 22, 30, 42 (S1..S4).
- MDP: states Light/Heavy, actions Extend Green/Switch, gamma = 0.9. Rewards: (Light, Extend) = -2, (Light, Switch) = +4, (Heavy, Extend) = +8, (Heavy, Switch) = -5. Transitions are in the `TRANS` table in the code.
- Repeated readings are treated as independent, and each posterior becomes the next prior.

## Sample input and output (`python src/traffic_controller.py`)
```
=== MINI TRAFFIC LIGHT CONTROLLER ===
Assumptions: prior P(Heavy)=0.30; sensor P(High|Heavy)=0.85, P(High|Light)=0.20

--- 1. BAYESIAN INFERENCE ---
Input : prior P(Heavy)=0.300, readings=['High']
Step  : reading High: P(Heavy|High) = 0.85*0.300 / 0.3950 = 0.6456
Output: P(Heavy traffic) = 64.6%
Why   : a noisy vehicle count is turned into a probability instead of a hard yes/no.

--- 2. LOCAL SEARCH (hill climbing, minimize expected delay) ---
Input : 4 settings (main/side green): S1=20/40s, S2=30/30s, S3=40/20s, S4=50/10s
Step  : S1 expected delay = 0.646*70 + 0.354*30 = 55.82 s
Step  : S2 expected delay = 0.646*52 + 0.354*22 = 41.37 s
Step  : S3 expected delay = 0.646*40 + 0.354*30 = 36.46 s
Step  : S4 expected delay = 0.646*46 + 0.354*42 = 44.58 s
Search: start S1 -> S2 -> S3 (stops: no neighbor is better)
Output: selected timing S3 (Main 40s / Side 20s), delay 36.46 s/vehicle
Why   : it walks to the neighboring timing with lower expected delay under the Bayesian belief.
        (Without the sensor, prior alone would pick S2.)

--- 3. MDP (Extend Green vs Switch) ---
Input : states Light/Heavy, rewards R(s,a), transitions P(s'|s,a), gamma=0.9
Step  : one-step Q(Extend Green) = 0.646*8 + 0.354*-2 = 4.46
Step  : one-step Q(Switch) = 0.646*-5 + 0.354*4 = -1.81
        Bellman (value iteration) Q(Extend Green) = 53.36
        Bellman (value iteration) Q(Switch) = 47.62
Output: decision = EXTEND GREEN
Why   : it picks the action with the larger belief-weighted long-run reward.
        (Prior alone would choose Switch.)

=== FINAL SIGNAL PLAN ===
Timing S3: Main 40s / Side 20s | P(Heavy)=64.6% | Decision: Extend Green

=== EVALUATION vs fixed 30/30 timing (S2) ===
reading   P(Heavy)  chosen   delay  fixed S2   saved
High         64.6%      S3   36.46     41.37    4.91
Low           7.4%      S2   24.23     24.23    0.00
```

## Explanation of each algorithm
1. **Bayes' rule:** P(Heavy|High) = P(High|Heavy)P(Heavy) / P(High). Chosen because the sensor is noisy: a "High" reading raises the belief from 30% to 64.6% without claiming certainty.
2. **Hill climbing:** moves to the neighboring timing with lower expected delay and stops when no neighbor is better. Chosen because the settings form an ordered line and the answer must be cheap to compute. Here it reaches S3, which is also the global best.
3. **MDP:** Q(s,a) = R(s,a) + gamma * sum P(s'|s,a) V(s'), computed by value iteration, then weighted by the belief. Chosen because "extend or switch" affects future traffic, not only the next second. The one-step reward is shown too.

## Results
With a "High" reading the chosen timing cuts expected delay from 41.37 s (fixed 30/30) to 36.46 s, a saving of 4.91 s per vehicle. With a "Low" reading it keeps 30/30 and switches, saving nothing, as it should. Limits: all numbers are assumed, and the single intersection is a toy model.

## AI-assisted code
The program structure and report formatting were drafted with Claude (an AI assistant), then checked by the author against the course material. The author can explain every formula above.

## Possible improvements
Learn the sensor and delay tables from real counts; use an HMM so traffic evolves over time (Lecture 7); use Q-learning instead of a known transition table (Lecture 5); add more roads and use A* or beam search over phase plans.

## Likely defense questions
- **Why is the Bayesian step needed?** It converts a noisy count into a probability that both other algorithms use.
- **Can hill climbing get stuck?** Yes, in a local minimum. Here the delay curve over S1..S4 has one minimum; with more settings use random restarts or annealing.
- **Why does the decision change with evidence?** The prior (30%) favors Switch; at 64.6% Heavy, Extend Green has the larger expected reward.
- **What is a Bellman update?** Reward now plus discounted value of the next state, averaged over transitions.
- **What if the sensor said "Low"?** P(Heavy) falls to 7.4%, so it keeps 30/30 and switches.

## Repository structure
```
src/traffic_controller.py   the three algorithms and the report
web/index.html             interactive website (same model, runs in the browser)
tests/test_controller.py   13 unit tests
docs/sample_output.txt     output of the default run
AI_USAGE.md                AI usage declaration
CONTRIBUTIONS.md           contribution statement
docs/DEMO_SCRIPT.md        script for the demo video
```

## Tests
```
python -m unittest discover -s tests -v
```
The tests check the Bayes posteriors (64.6% after "High", 7.4% after "Low"), that hill climbing matches a brute-force search for every belief from 0% to 100%, that it is never worse than the fixed 30/30 timing, that value iteration satisfies the Bellman equation, and that the sensor evidence changes the chosen timing.
