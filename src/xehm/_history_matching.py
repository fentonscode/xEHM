__all__ = ["HistoryMatching"]


# Overall history match object to use from within Python (rather than the command line)
class HistoryMatching:
    def __init__(self):
        self.build_emulator = None
        self.out_space = None
        self.input_variables = []

    def load(self, f_name: str):
        pass

    def save(self, f_name: str):
        pass

    def set_input_dimensions(self):
        pass