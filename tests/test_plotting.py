from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from matplotlib.axes import Axes

from sycophancy_testbed.plotting import plot_trajectory


class TrajectoryPlotTests(unittest.TestCase):
    def test_each_plot_contains_only_its_speakers_scores(self):
        df = pd.DataFrame({"turn": [1, 2, 3], "user_H": [4, 2, 1], "assistant_H": [-2, -3, -4]})
        series = []
        original = Axes.plot

        def capture(ax, x, y, **kwargs):
            series.append((list(x), list(y)))
            return original(ax, x, y, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(Axes, "plot", capture):
            paths = plot_trajectory(df, str(Path(directory) / "trajectory.png"))
            self.assertEqual([Path(p).name for p in paths], ["trajectory_user.png", "trajectory_assistant.png"])
            for path in paths:
                self.assertTrue(Path(path).read_bytes().startswith(b"\x89PNG"))
        self.assertEqual(series, [([1, 2, 3], [4, 2, 1]), ([1, 2, 3], [-2, -3, -4])])
