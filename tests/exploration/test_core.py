import json
import tempfile
import unittest

import numpy as np

from exploration.events import Event, Option, Prefix, Tail
from exploration.feedback import Feedback, TrackedFrame
from exploration.memory import ScalarMemory, StructuredMemory, means
from exploration.protocol import Protocol, rng_for
from exploration.selectors import select, voi
from exploration.trials import PrefixResult, TailResult, TrialManager


class MemoryTests(unittest.TestCase):
    def test_prefix_failure_shares_across_modes_without_tail_update(self):
        m = StructuredMemory()
        before, tail = m.scores().reshape(2, 3), m.gamma.copy()
        m.update(Event(Option(0, 0), Prefix.NOT_READY))
        after = m.scores().reshape(2, 3)
        np.testing.assert_array_less(after[0], before[0])
        np.testing.assert_array_equal(after[1], before[1])
        np.testing.assert_array_equal(m.gamma, tail)
        self.assertEqual(select(m, "greedy").index, 3)

    def test_ready_unhelpful_tail_preserves_grasp_evidence(self):
        m = StructuredMemory()
        m.update(Event(Option(0, 0), Prefix.READY, Tail.AWAY))
        self.assertGreater(means(m.alpha)[0, Prefix.READY], .25)
        score = m.scores().reshape(2, 3)
        self.assertLess(score[0, 0], score[0, 1])
        np.testing.assert_array_equal(m.gamma[0, 1:], np.full((2, 6), 1/6))

    def test_unknown_is_distinct(self):
        m = StructuredMemory()
        m.update(Event(Option(0, 0), Prefix.UNKNOWN))
        self.assertEqual(m.alpha[0, Prefix.NOT_READY], .25)
        self.assertEqual(m.alpha[0, Prefix.UNKNOWN], 1.25)
        m.update(Event(Option(0, 0), Prefix.READY, Tail.UNKNOWN))
        self.assertAlmostEqual(m.gamma[0, 0, Tail.UNKNOWN], 1 + 1/6)
        self.assertAlmostEqual(m.gamma[0, 0, Tail.NO_RESPONSE], 1/6)

    def test_44_parameters_and_no_privileged_record(self):
        m = StructuredMemory()
        self.assertEqual(m.alpha.size + m.gamma.size, 44)
        with self.assertRaises(ValueError):
            Event.from_record(dict(grasp=0, mode=0, prefix="unknown", tail=None, success=True))
        with self.assertRaises(ValueError):
            Event(Option(0, 0), Prefix.NOT_READY, Tail.NO_RESPONSE)

    def test_nine_branches_and_martingale(self):
        m = StructuredMemory()
        m.update(Event(Option(1, 2), Prefix.READY, Tail.TOWARD))
        for c in m.options:
            branches = m.branches(c)
            self.assertEqual(len(branches), 9)
            self.assertAlmostEqual(sum(p for p, _ in branches), 1.)
            expected = np.zeros(6)
            for prob, e in branches:
                updated = m.copy()
                updated.update(e)
                self.assertAlmostEqual(updated.alpha.sum() - m.alpha.sum(), 1)
                self.assertAlmostEqual(updated.gamma.sum() - m.gamma.sum(), int(e.tail is not None))
                expected += prob * updated.scores()
            np.testing.assert_allclose(expected, m.scores(), atol=1e-14)

    def test_voi_does_not_mutate_and_single_option_is_zero(self):
        m = StructuredMemory()
        before = m.snapshot()
        self.assertTrue((voi(m) >= 0).all())
        self.assertEqual(m.snapshot(), before)
        single = StructuredMemory(1, 1)
        self.assertAlmostEqual(voi(single)[0], 0, places=12)
        self.assertAlmostEqual(voi(ScalarMemory(single))[0], 0, places=12)

    def test_scalar_prior_means_match_smoothed_structured_prior(self):
        m = StructuredMemory.fit_prior([Event(Option(0, 1), Prefix.READY, Tail.TOWARD)] * 5)
        for strength in (1., 3., 6.):
            scalar = ScalarMemory(m, strength)
            np.testing.assert_allclose(scalar.scores(), m.scores(), atol=1e-14)
            np.testing.assert_allclose(scalar.counts.sum(-1), strength)
        self.assertAlmostEqual(m.alpha[0].sum(), 3.)
        self.assertAlmostEqual(m.alpha[1].sum(), 1.)

    def test_voi_can_choose_less_immediate_reward(self):
        # A strong incumbent and an uncertain alternative: inspectable fixed fixture.
        m = StructuredMemory(2, 1,
            alpha=[[.01, 100, .01, .01], [.01, 1, .01, .01]],
            gamma=[[[20, 20, 20, 20, 100, 20]], [[.05, .05, .05, .05, .15, .05]]])
        greedy, probe = select(m, "greedy"), select(m, "voi")
        self.assertEqual(greedy.index, 0)
        self.assertEqual(probe.index, 1)
        self.assertLess(m.scores()[1], m.scores()[0])

    def test_same_scalar_history_different_explanation(self):
        miss, away = StructuredMemory(), StructuredMemory()
        a, b = Event(Option(0, 0), Prefix.NOT_READY), Event(Option(0, 0), Prefix.READY, Tail.AWAY)
        miss.update(a)
        away.update(b)
        self.assertNotEqual(miss.snapshot(), away.snapshot())
        sa, sb = ScalarMemory(StructuredMemory()), ScalarMemory(StructuredMemory())
        sa.update(a)
        sb.update(b)
        np.testing.assert_array_equal(sa.counts, sb.counts)


def f(x=0., handle=0., opening=0., confidence=1., gripper=.3):
    return TrackedFrame((x, 0., 0.), gripper,
                        None if handle is None else (handle, 0., 0.), opening, confidence)


class FeedbackTests(unittest.TestCase):
    def test_goal_before_slip_or_failed_grasp(self):
        detector = Feedback()
        self.assertEqual(detector.prefix([f(gripper=1)], [f(.1, 0, 1.1)]), Prefix.GOAL)
        self.assertEqual(detector.tail([f(), f(.1, 0, 1.1)]), Tail.GOAL)

    def test_closure_alone_is_not_readiness(self):
        detector = Feedback()
        self.assertEqual(detector.prefix([f(gripper=1)], [f(), f()]), Prefix.UNKNOWN)
        self.assertEqual(detector.prefix([f(gripper=1)], [f(), f(.006, .006)]), Prefix.READY)
        self.assertEqual(detector.prefix([f(gripper=1)], [f(), f(.1, 0.)]), Prefix.NOT_READY)

    def test_missingness_is_not_zero_motion(self):
        detector = Feedback()
        self.assertEqual(detector.tail([f(), f(handle=None, opening=None)]), Tail.UNKNOWN)
        self.assertEqual(detector.tail([f(), f(opening=None)]), Tail.UNKNOWN)
        self.assertEqual(detector.tail([f(), f()]), Tail.NO_RESPONSE)
        self.assertEqual(detector.tail([f(), f(opening=.1)]), Tail.TOWARD)
        self.assertEqual(detector.tail([f(opening=.2), f()]), Tail.AWAY)


class RecordingBackend:
    def __init__(self, prefix=Prefix.READY):
        self.prefix = prefix
        self.resets = []
        self.tail_calls = 0

    def reset(self, scene, seed):
        self.resets.append((scene, seed))

    def execute_prefix(self, grasp, rng):
        return PrefixResult(self.prefix, 4, str((grasp, rng.random())), {})

    def execute_mode(self, option, rng):
        self.tail_calls += 1
        return TailResult(Tail.UNKNOWN, 5, {})

    def evaluation(self):
        return {"simulator_success": False, "privileged_access": ["test fixture only"]}

    def close(self):
        pass


class TrialTests(unittest.TestCase):
    def test_prefix_identity_mode_independent(self):
        backend = RecordingBackend()
        trials = TrialManager(backend, Protocol())
        a = trials.attempt(7, 0, 0, Option(0, 0))[1]
        b = trials.attempt(7, 0, 0, Option(0, 2))[1]
        self.assertEqual(a["prefix_fingerprint"], b["prefix_fingerprint"])
        c = trials.attempt(7, 0, 1, Option(0, 0))[1]
        self.assertNotEqual(a["prefix_fingerprint"], c["prefix_fingerprint"])

    def test_unknown_prefix_never_executes_tail(self):
        backend = RecordingBackend(Prefix.UNKNOWN)
        trials = TrialManager(backend, Protocol())
        event, record, _ = trials.attempt(7, 0, 0, Option(0, 0))
        self.assertEqual(backend.tail_calls, 0)
        self.assertEqual(record["tail_steps"], 0)
        self.assertIsNone(event.tail)

    def test_active_final_does_not_update_and_truth_separate(self):
        trials = TrialManager(RecordingBackend(), Protocol())
        with tempfile.TemporaryDirectory() as out:
            online, truth = trials.active("structured_voi", StructuredMemory(), 7, 0, out)
            self.assertEqual(len(online), 4)
            self.assertEqual(online[-1]["memory_before"], online[-1]["memory_after"])
            self.assertNotIn("simulator_success", json.dumps(online))
            self.assertIn("simulator_success", truth[0])

    def test_protocol_disjoint_budgets(self):
        self.assertEqual(Protocol().record()["budgets"], {"train": 540, "active": 1800, "common_history": 180})
        with self.assertRaises(ValueError):
            Protocol(dev=(210000,))


if __name__ == "__main__":
    unittest.main()
