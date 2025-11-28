import numpy as np

class OutputAdapter:
    def adapt_output(self, y):
        y = np.array(y, dtype=float)
        return y.flatten()  # <--- ADD THIS
