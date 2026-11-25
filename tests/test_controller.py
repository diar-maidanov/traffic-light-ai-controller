import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import traffic_controller as tc

class TestBayes(unittest.TestCase):
    def test_high_reading_posterior(self):
        p, _ = tc.bayes_update(0.30, "High")
        self.assertAlmostEqual(p, 0.6456, places=3)
    def test_low_reading_lowers_belief(self):
        p, _ = tc.bayes_update(0.30, "Low")
        self.assertLess(p, 0.30)
        self.assertAlmostEqual(p, 0.0744, places=3)
    def test_repeated_high_raises_belief(self):
        p1, _ = tc.bayes_update(0.30, "High"); p2, _ = tc.bayes_update(p1, "High")
        self.assertGreater(p2, p1)
    def test_posterior_is_probability(self):
        for prior in (0.01, 0.5, 0.99):
            for o in ("High", "Low"):
                self.assertTrue(0 <= tc.bayes_update(prior, o)[0] <= 1)

class TestLocalSearch(unittest.TestCase):
    def test_high_selects_s3(self):
        self.assertEqual(tc.hill_climb(tc.bayes_update(0.30, "High")[0])[0], 2)
    def test_low_selects_s2(self):
        self.assertEqual(tc.hill_climb(tc.bayes_update(0.30, "Low")[0])[0], 1)
    def test_matches_brute_force_on_grid(self):
        for k in range(0, 101):
            p = k / 100
            best = min(range(len(tc.SETTINGS)), key=lambda i: tc.expected_delay(i, p))
            self.assertEqual(tc.hill_climb(p)[0], best, f"p={p}")
    def test_never_worse_than_fixed(self):
        for k in range(0, 101):
            p = k / 100
            self.assertLessEqual(tc.expected_delay(tc.hill_climb(p)[0], p), tc.expected_delay(1, p) + 1e-9)

class TestMDP(unittest.TestCase):
    def test_value_iteration_satisfies_bellman(self):
        V = tc.value_iteration()
        for s in tc.STATES:
            self.assertAlmostEqual(V[s], max(tc.q_value(s, a, V) for a in tc.ACTIONS), places=2)  # 100 iterations, gamma=0.9
    def test_heavy_belief_extends(self):
        q = tc.belief_q(0.9, tc.value_iteration())
        self.assertEqual(max(q, key=q.get), "Extend Green")
    def test_light_belief_switches(self):
        q = tc.belief_q(0.0, tc.value_iteration())
        self.assertEqual(max(q, key=q.get), "Switch")
    def test_one_step_values_for_high_reading(self):
        q = tc.belief_q(tc.bayes_update(0.30, "High")[0], {s: 0 for s in tc.STATES})
        self.assertAlmostEqual(q["Extend Green"], 4.456, places=2)
        self.assertAlmostEqual(q["Switch"], -1.81, places=2)

class TestPipeline(unittest.TestCase):
    def test_evidence_changes_decision(self):
        prior_t = tc.decide(0.30)[0]; post_t = tc.decide(tc.bayes_update(0.30, "High")[0])[0]
        self.assertNotEqual(prior_t, post_t)

if __name__ == "__main__":
    unittest.main(verbosity=2)
