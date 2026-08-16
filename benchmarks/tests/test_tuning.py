from __future__ import annotations

import math
import unittest

from benchmarks.tuning import (
    Candidate,
    CandidateResult,
    TuningError,
    rank_candidates,
    score_samples,
    sequential_halving,
)


class ScoreTests(unittest.TestCase):
    def test_score_uses_linear_p5_and_sample_cv(self) -> None:
        score = score_samples(Candidate("alpha", {"threads": 6}), [10, 20, 30], 128)

        self.assertAlmostEqual(score.p5, 11.0)
        self.assertAlmostEqual(score.mean, 20.0)
        self.assertAlmostEqual(score.median, 20.0)
        self.assertAlmostEqual(score.cv, 0.5)
        self.assertEqual(score.sample_count, 3)

    def test_rank_prefers_p5_and_marks_high_cv_unstable(self) -> None:
        stable = CandidateResult(Candidate("stable", {}), (12.0, 12.0, 12.0), 256)
        bursty = CandidateResult(Candidate("bursty", {}), (11.0, 30.0, 30.0), 256)

        ranked = rank_candidates([bursty, stable], max_cv=0.20)

        self.assertEqual([score.candidate.candidate_id for score in ranked], ["stable", "bursty"])
        self.assertTrue(ranked[0].stable)
        self.assertFalse(ranked[1].stable)

    def test_invalid_samples_return_structured_error(self) -> None:
        for samples in ([], [0.0], [-1.0], [True], [math.nan], [math.inf]):
            with self.subTest(samples=samples):
                with self.assertRaises(TuningError) as caught:
                    score_samples(Candidate("bad", {}), samples, 1)
                payload = caught.exception.to_dict()
                self.assertFalse(payload["success"])
                self.assertEqual(payload["error"]["code"], "invalid_samples")


class SequentialHalvingTests(unittest.TestCase):
    def test_evaluates_all_then_only_survivors(self) -> None:
        candidates = [Candidate(name, {"name": name}) for name in "abcd"]
        calls: list[tuple[str, int]] = []
        samples = {
            ("a", 128): (10.0, 10.1, 10.2),
            ("b", 128): (13.0, 13.1, 13.2),
            ("c", 128): (9.0, 9.1, 9.2),
            ("d", 128): (12.0, 12.1, 12.2),
            ("b", 512): (15.0, 15.1, 15.2),
            ("d", 512): (16.0, 16.1, 16.2),
        }

        def evaluate(candidate: Candidate, budget: int) -> tuple[float, ...]:
            calls.append((candidate.candidate_id, budget))
            return samples[(candidate.candidate_id, budget)]

        result = sequential_halving(candidates, [128, 512], evaluate, reduction_factor=2)

        self.assertEqual(calls[:4], [(name, 128) for name in "abcd"])
        self.assertEqual(set(calls[4:]), {("b", 512), ("d", 512)})
        self.assertEqual(result.winner.candidate.candidate_id, "d")
        self.assertEqual(
            [candidate.candidate_id for candidate in result.rounds[0].survivors],
            ["b", "d"],
        )
        self.assertEqual(result.rounds[1].budget, 512)

    def test_rejects_duplicate_candidates_and_invalid_schedule(self) -> None:
        candidate = Candidate("same", {})
        cases = [
            ((candidate, candidate), (128, 512), 2, "duplicate_candidate"),
            ((candidate,), (), 2, "invalid_budgets"),
            ((candidate,), (512, 128), 2, "invalid_budgets"),
            ((candidate,), (128,), 1, "invalid_reduction_factor"),
        ]
        for candidates, budgets, factor, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(TuningError) as caught:
                    sequential_halving(candidates, budgets, lambda _c, _b: (1.0,), factor)
                self.assertEqual(caught.exception.code, code)

    def test_wraps_evaluator_failure_with_candidate_and_budget(self) -> None:
        def fail(_candidate: Candidate, _budget: int) -> tuple[float, ...]:
            raise RuntimeError("process exited")

        with self.assertRaises(TuningError) as caught:
            sequential_halving([Candidate("broken", {})], [128], fail)

        self.assertEqual(caught.exception.code, "evaluation_failed")
        self.assertEqual(caught.exception.details["candidate_id"], "broken")
        self.assertEqual(caught.exception.details["budget"], 128)

    def test_candidate_inputs_are_validated(self) -> None:
        with self.assertRaises(TuningError) as caught:
            Candidate(" ", {})
        self.assertEqual(caught.exception.code, "invalid_candidate")

        with self.assertRaises(TuningError):
            Candidate("x", {"threads": object()})


if __name__ == "__main__":
    unittest.main()
