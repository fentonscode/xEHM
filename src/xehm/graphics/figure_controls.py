import matplotlib.pyplot as plt
import matplotlib.gridspec as grid

__all__ = ["HGraph", "side_by_side", "single_figure", "vertical_split_percent"]


# Wrapper for matplotlib bits and pieces
class HGraph:
    def __init__(self):
        self.fig = None
        self.axes = None

    def set_dimensions(self, rows: int, cols: int):
        self.fig = plt.figure(figsize=(10, 10))
        gs = grid.GridSpec(rows, cols)
        self.axes = [self.fig.add_subplot(gs[i]) for i in range(rows * cols)]
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