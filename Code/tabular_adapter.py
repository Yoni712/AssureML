# adapters/tabular_adapter.py

import numpy as np

class TabularAdapter:
    def adapter_input(self, input_list):
        """
        Expects input like [5.1, 3.5, 1.4, 0.2]
        Outputs shape (1, 4)
        """
        arr = np.array(input_list, dtype=float)
        return arr.reshape(1, -1)
