import numpy as np
import pandas as pd
import performanceMeasure as pm

pm.performanceMeasurePlot(pd.read_csv("funnel10.csv"),plotfn="funnel10.pdf" )
pm.performanceMeasurePlot(pd.read_csv("AR1model.csv"),plotfn="AR1model.pdf" )
pm.performanceMeasurePlot(pd.read_csv("SWmodel.csv"),plotfn="SWmodel.pdf" )