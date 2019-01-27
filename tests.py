'''
Tests to Run on TRAVIS CI
'''
import sys, os
import pandas as pd
from glob import glob
import numpy as np
import time
from imageCl import genImageCl
#from tsp-local_2opt import Two_opt

if __name__=="__main__":
	print("Image Classification Tests, Run Only using TF and Create imageCl Object using h5paths and dataset Path")