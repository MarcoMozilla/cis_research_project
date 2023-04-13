import numpy as np


class SphereMark:

    def __init__(self, data_BxF):
        self.data_BxF = data_BxF
        self.B, self.F = self.data_BxF

        self.arc = np.pi / 4
