import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as opt
from stacking_functions import *

IX = ["2","27","3"]

dfs = {ix: pd.read_csv(f"mod-nopiv-stacked-ind{ix}-onecrab-results.csv") for ix in IX}
PhoebePlotting.SpectrumPlots(dfs, "crab_sepctrum")
PhoebePlotting.LikelihoodPlots(dfs, "crab_lh")
