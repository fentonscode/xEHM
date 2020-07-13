import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from numpy import ceil, sqrt

__all__ = ["HGraph", "side_by_side", "single_figure", "vertical_split_percent", "squared_panel"]


# Wrapper for matplotlib bits and pieces
class HGraph:
    def __init__(self):
        self.fig = None
        self.axes = None

    def set_dimensions(self, rows: int, cols: int, extend: bool = False):
        fig_width = 10
        fig_height = 10
        if extend:
            fig_width *= cols
            fig_height *= rows
        self.fig = plt.figure(figsize=(fig_width, fig_height))
        gs = grid.GridSpec(rows, cols)
        self.axes = [self.fig.add_subplot(gs[int(i)]) for i in range(rows * cols)]
        return self

    def plot(self):
        self.fig.show()

    def send_to_file(self, f_name: str):
        raise NotImplementedError("Coming soon")


def single_figure() -> HGraph:
    graph = HGraph()
    graph.set_dimensions(1, 1)
    return graph


def side_by_side() -> HGraph:
    graph = HGraph()
    graph.set_dimensions(1, 2)
    return graph


def vertical_split_percent(percent: float) -> HGraph:
    lower = int(percent * 100)
    upper = 100 - lower
    graph = HGraph()
    graph.fig = plt.figure(figsize=(10, 10))
    graph.axes = []
    graph.axes.append(plt.subplot2grid((100, 1), (0, 0), rowspan=upper, fig=graph.fig))
    graph.axes.append(plt.subplot2grid((100, 1), (upper, 0), rowspan=lower, fig=graph.fig))
    return graph


# Create a square panel with enough room for n_items charts.
# Biased towards being wide to suit standard display aspect ratios
def squared_panel(n_items: int) -> HGraph:
    graph = HGraph()
    width: int = ceil(sqrt(n_items))
    height: int = (n_items + width - 1) // width
    graph.set_dimensions(int(height), int(width), extend=True)
    return graph
