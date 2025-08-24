import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(parent_dir)
from Preprocessing import Preprocessing 
from SignalRefinement import SignalRefinement 
from SampleConversion import SampleConversion

if __name__ == "__main__":
    directory = "Dataset_NeckPass"
    Preprocessing(directory)    # Preprocessing
    SignalRefinement(directory)    # Signal Refinement. The result will be saved in str(directory + "_filtered")
    SampleConversion(directory + "_filtered")   # Sample Conversion. The result will be saved in str(directory + "_filtered" + "_extract")
