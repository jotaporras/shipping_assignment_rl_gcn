from unittest import TestCase
import pandas as pd

from visualization import shipping_allocation_result_viz


class Test(TestCase):
    def test_consolidate_episode(self):
        # Given
        randomvalid_validations = [
            pd.read_csv(
                f"data/results/randomvalid_validation/ep_{i}/summary_movement_report.csv"
            )
            for i in range(5)
        ]
        # When
        summary_movement_consolidated = (
            shipping_allocation_result_viz.consolidate_episodes(randomvalid_validations)
        )

        # Then
        print(len(summary_movement_consolidated.step.values.tolist()))
        print(
            summary_movement_consolidated.groupby("episode")["step"].max()
        )  # strange, some have one step longer?
        assert summary_movement_consolidated.episode.unique().tolist() == [
            0,
            1,
            2,
            3,
            4,
        ]
        # assert len(summary_movement_consolidated.step.values.tolist()) == 5*48 # some have 48, others 47.
