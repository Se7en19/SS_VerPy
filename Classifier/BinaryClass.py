'''This file is the main file for the binary classifier for the iris dataset'''

import pandas as pd
import numpy as np 

class Classifier:

    def __init__(self):
        self.X, self.y, self.w, self.eta, self.numEpochs = None

    