import pandas as pd
import matplotlib.pylab as pl
import numpy as np
import os

if __name__ == '__main__':

	data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'MindTheGap-1.0-Data.tsv'), header=0, \
                    delimiter="\t", quoting=3)

	print "Number of question samples: "
	print data.shape
	print data.head

