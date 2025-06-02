import unittest

import numpy as np
import pytest
from rl_exercises.week_3 import EpsilonGreedyPolicy, SARSAAgent


class TestSARSA(unittest.TestCase):
    def test_td_update_computation(self):
        """Test that update_agent(...) implements the SARSA TD‐error correctly."""

        # Dummy env with 2 actions
        class DummyEnv:
            action_space = type("A", (), {"n": 2})()

        env = DummyEnv()
        policy = EpsilonGreedyPolicy(env, epsilon=0.0, seed=0)
        agent = SARSAAgent(env, policy, alpha=0.5, gamma=0.9)

        # Manually seed Q so we know what should happen
        agent.Q[0][1] = 2.0
        agent.Q[1][0] = 0.5

        # Perform one update: (s=0,a=1,r=1.0,s'=1,a'=0,done=False)
        new_q = agent.update_agent(
            state=0,
            action=1,
            reward=1.0,
            next_state=1,
            next_action=0,
            done=False,
        )

        # δ = α * [(r + γ Q[s',a']) - Q[s,a]]
        expected_delta = 0.5 * ((1.0 + 0.9 * 0.5) - 2.0)
        expected_new_q = 2.0 + expected_delta

        self.assertAlmostEqual(new_q, expected_new_q)
        self.assertAlmostEqual(agent.Q[0][1], expected_new_q)

    def test_predict_action_greedy_vs_random(self):
        """Test ε=0 is greedy, ε=1 is (reproducibly) random."""

        class DummyEnv:
            action_space = type("A", (), {"n": 5})()

        env = DummyEnv()

        # ε = 0 → always pick argmax
        policy0 = EpsilonGreedyPolicy(env, epsilon=0.0, seed=42)
        agent0 = SARSAAgent(env, policy0, alpha=0.1, gamma=0.9)
        agent0.Q["s"] = np.array([1, 1, 5, 1, 1])
        picks0 = {agent0.predict_action("s", evaluate=False) for _ in range(20)}
        self.assertEqual(picks0, {2})

        # ε = 1 → uniform random (via the policy’s own RNG)
        policy1 = EpsilonGreedyPolicy(env, epsilon=1.0, seed=123)
        agent1 = SARSAAgent(env, policy1, alpha=0.1, gamma=0.9)
        picks1 = [agent1.predict_action("s", evaluate=False) for _ in range(200)]
        self.assertGreaterEqual(len(set(picks1)), 4)

    def test_terminal_state_ignores_next_q(self):
        """Test that when done=True, the next-Q term is zeroed out."""

        class DummyEnv:
            action_space = type("A", (), {"n": 2})()

        env = DummyEnv()
        policy = EpsilonGreedyPolicy(env, epsilon=0.0, seed=0)
        agent = SARSAAgent(env, policy, alpha=0.5, gamma=0.9)

        # Set Q so stale next‐state value is obviously wrong
        agent.Q[0][1] = 1.0
        agent.Q[1][0] = 100.0

        new_q = agent.update_agent(
            state=0,
            action=1,
            reward=1.0,
            next_state=1,
            next_action=0,
            done=True,
        )

        # Correct behavior: ignore Q[1][0], so new_q == old_q == 1.0
        assert new_q == pytest.approx(1.0)
        assert agent.Q[0][1] == pytest.approx(1.0)


if __name__ == "__main__":
    unittest.main()
