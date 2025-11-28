import numpy as np

class DataAdapter:
    def adapt_input(self, x):
        return np.array(x, dtype=float)
