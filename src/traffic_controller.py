"""
Mini Traffic Light Controller (Final Project, topic 08)
One intersection, 2 roads (Main, Side), 4 signal settings, 2 hidden traffic states.

Three course algorithms, ONE shared goal (a smarter signal):
  1. Bayesian inference : vehicle sensor reading -> P(Heavy traffic on Main)
  2. Local search       : hill climbing over 4 timing settings, scored with that belief
  3. MDP                : Extend Green vs Switch, Q-values weighted by that belief
The Bayesian belief is the shared input: it changes BOTH the chosen timing and the decision.

All numbers below are labeled ASSUMPTIONS (hypothetical, no dataset needed).
Usage:  python traffic_controller.py [--obs High Low ...] [--prior 0.30]
"""
import argparse

# ---------- ASSUMPTIONS ----------
PRIOR_HEAVY = 0.30                                   # P(Heavy) before any reading
SENSOR = {"Heavy": {"High": 0.85, "Low": 0.15},      # P(reading | true state)
          "Light": {"High": 0.20, "Low": 0.80}}
SETTINGS = [("S1", 20, 40), ("S2", 30, 30), ("S3", 40, 20), ("S4", 50, 10)]  # (name, main green s, side green s)
DELAY = {"Heavy": [70, 52, 40, 46],                  # avg delay (s/vehicle) of each setting per traffic state
         "Light": [30, 22, 30, 42]}
STATES, ACTIONS = ["Light", "Heavy"], ["Extend Green", "Switch"]
REWARD = {("Light", "Extend Green"): -2, ("Light", "Switch"): 4,    # R(s, a)
          ("Heavy", "Extend Green"): 8, ("Heavy", "Switch"): -5}
TRANS = {("Light", "Extend Green"): {"Light": .7, "Heavy": .3},     # P(s' | s, a)
         ("Light", "Switch"):       {"Light": .8, "Heavy": .2},
         ("Heavy", "Extend Green"): {"Light": .4, "Heavy": .6},
         ("Heavy", "Switch"):       {"Light": .2, "Heavy": .8}}
GAMMA = 0.9


# ---------- 1. BAYESIAN INFERENCE ----------
def bayes_update(p_heavy, obs):
    like_h, like_l = SENSOR["Heavy"][obs], SENSOR["Light"][obs]
    evidence = like_h * p_heavy + like_l * (1 - p_heavy)
    return like_h * p_heavy / evidence, evidence


# ---------- 2. LOCAL SEARCH (hill climbing, minimize expected delay) ----------
def expected_delay(i, p_heavy):
    return p_heavy * DELAY["Heavy"][i] + (1 - p_heavy) * DELAY["Light"][i]

def hill_climb(p_heavy, start=0):
    cur, path = start, [start]
    while True:
        nbrs = [j for j in (cur - 1, cur + 1) if 0 <= j < len(SETTINGS)]
        best = min(nbrs, key=lambda j: expected_delay(j, p_heavy))
        if expected_delay(best, p_heavy) >= expected_delay(cur, p_heavy):
            return cur, path                          # no better neighbor: stop
        cur = best
        path.append(cur)


# ---------- 3. MDP (Bellman backup / value iteration) ----------
def q_value(s, a, V):
    return REWARD[(s, a)] + GAMMA * sum(p * V[s2] for s2, p in TRANS[(s, a)].items())

def value_iteration(iters=100):
    V = {s: 0.0 for s in STATES}
    for _ in range(iters):
        V = {s: max(q_value(s, a, V) for a in ACTIONS) for s in STATES}
    return V

def belief_q(p_heavy, V):
    b = {"Heavy": p_heavy, "Light": 1 - p_heavy}
    return {a: sum(b[s] * q_value(s, a, V) for s in STATES) for a in ACTIONS}


# ---------- pipeline + report ----------
def decide(p_heavy):
    best, path = hill_climb(p_heavy)
    V0 = {s: 0.0 for s in STATES}
    q1 = belief_q(p_heavy, V0)                        # one-step expected reward
    qn = belief_q(p_heavy, value_iteration())         # long-run (Bellman) value
    return best, path, q1, qn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", nargs="+", default=["High"], choices=["High", "Low"],
                    help="vehicle-sensor readings, applied in order")
    ap.add_argument("--prior", type=float, default=PRIOR_HEAVY)
    args = ap.parse_args()

    print("=== MINI TRAFFIC LIGHT CONTROLLER ===")
    print(f"Assumptions: prior P(Heavy)={args.prior:.2f}; sensor P(High|Heavy)=0.85, P(High|Light)=0.20")

    print("\n--- 1. BAYESIAN INFERENCE ---")
    print(f"Input : prior P(Heavy)={args.prior:.3f}, readings={args.obs}")
    p = args.prior
    for o in args.obs:
        new, ev = bayes_update(p, o)
        print(f"Step  : reading {o}: P(Heavy|{o}) = {SENSOR['Heavy'][o]}*{p:.3f} / {ev:.4f} = {new:.4f}")
        p = new
    print(f"Output: P(Heavy traffic) = {p:.1%}")
    print("Why   : a noisy vehicle count is turned into a probability instead of a hard yes/no.")

    best, path, q1, qn = decide(p)
    base_best, base_path, base_q1, base_qn = decide(args.prior)

    print("\n--- 2. LOCAL SEARCH (hill climbing, minimize expected delay) ---")
    print(f"Input : 4 settings (main/side green): " + ", ".join(f"{n}={m}/{s}s" for n, m, s in SETTINGS))
    for i, (n, m, s) in enumerate(SETTINGS):
        print(f"Step  : {n} expected delay = {p:.3f}*{DELAY['Heavy'][i]} + {1-p:.3f}*{DELAY['Light'][i]} = {expected_delay(i, p):.2f} s")
    print("Search: start " + " -> ".join(SETTINGS[i][0] for i in path) + " (stops: no neighbor is better)")
    n, m, s = SETTINGS[best]
    print(f"Output: selected timing {n} (Main {m}s / Side {s}s), delay {expected_delay(best, p):.2f} s/vehicle")
    print(f"Why   : it walks to the neighboring timing with lower expected delay under the Bayesian belief.")
    print(f"        (Without the sensor, prior alone would pick {SETTINGS[base_best][0]}.)")

    print("\n--- 3. MDP (Extend Green vs Switch) ---")
    print("Input : states Light/Heavy, rewards R(s,a), transitions P(s'|s,a), gamma=0.9")
    for a in ACTIONS:
        print(f"Step  : one-step Q({a}) = {p:.3f}*{REWARD[('Heavy', a)]} + {1-p:.3f}*{REWARD[('Light', a)]} = {q1[a]:.2f}")
    for a in ACTIONS:
        print(f"        Bellman (value iteration) Q({a}) = {qn[a]:.2f}")
    action = max(qn, key=qn.get)
    base_action = max(base_qn, key=base_qn.get)
    print(f"Output: decision = {action.upper()}")
    print(f"Why   : it picks the action with the larger belief-weighted long-run reward.")
    print(f"        (Prior alone would choose {base_action}.)")

    print("\n=== FINAL SIGNAL PLAN ===")
    print(f"Timing {n}: Main {m}s / Side {s}s | P(Heavy)={p:.1%} | Decision: {action}")

    print("\n=== EVALUATION vs fixed 30/30 timing (S2) ===")
    print(f"{'reading':8}{'P(Heavy)':>10}{'chosen':>8}{'delay':>8}{'fixed S2':>10}{'saved':>8}")
    for o in ["High", "Low"]:
        pp, _ = bayes_update(args.prior, o)
        b, _, _, _ = decide(pp)
        d, f = expected_delay(b, pp), expected_delay(1, pp)
        print(f"{o:8}{pp:>10.1%}{SETTINGS[b][0]:>8}{d:>8.2f}{f:>10.2f}{f-d:>8.2f}")

if __name__ == "__main__":
    main()
