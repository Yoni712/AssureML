import numpy as np

class OutputAdapter:
    def adapt_output(self, y):
        return np.array(y, dtype=float)
