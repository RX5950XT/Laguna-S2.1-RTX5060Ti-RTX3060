import unittest

from laguna_deploy.benchmark import decode_statistics, validate_acceptance


class DecodeStatisticsTests(unittest.TestCase):
    def test_reports_low_percentile_not_only_average(self) -> None:
        result = decode_statistics([20.0, 21.0, 22.0, 23.0, 24.0])

        self.assertEqual(result["minimum_tps"], 20.0)
        self.assertAlmostEqual(result["p5_tps"], 20.2)
        self.assertEqual(result["median_tps"], 22.0)

    def test_requires_at_least_one_measurement(self) -> None:
        with self.assertRaises(ValueError):
            decode_statistics([])


class AcceptanceTests(unittest.TestCase):
    def test_rejects_exactly_five_percent_quality_loss(self) -> None:
        accepted, reasons = validate_acceptance(
            context_size=262144,
            decode_tps=[21.0] * 10,
            quality_losses={"perplexity": 0.05},
        )

        self.assertFalse(accepted)
        self.assertIn("perplexity", " ".join(reasons))

    def test_rejects_context_reduction_and_unstable_throughput(self) -> None:
        accepted, reasons = validate_acceptance(
            context_size=131072,
            decode_tps=[19.0, 19.9] + [25.0] * 8,
            quality_losses={"perplexity": 0.01},
        )

        self.assertFalse(accepted)
        self.assertGreaterEqual(len(reasons), 2)

    def test_accepts_only_when_every_gate_passes(self) -> None:
        accepted, reasons = validate_acceptance(
            context_size=262144,
            decode_tps=[20.5] * 10,
            quality_losses={"perplexity": 0.01, "coding": 0.02},
        )

        self.assertTrue(accepted)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
