import pandas as pd
import matplotlib.pylab as pl
import numpy as np
import os
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


if __name__ == '__main__':

	
	data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'MindTheGap-1.0-Data.tsv'), header=0, \
                    delimiter="\t", quoting=3)
	dt1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'token_ct_features.csv'))
	dt2 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'lexical_feature.csv'))
	dt3 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'syntactic_feature.csv'))
	dt4 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'semantic_feature.csv'))

	'''
	print dt1.dtypes
	print
	print dt2.dtypes
	print
	print dt3.dtypes
	print
	print dt4.dtypes
	'''

	dtf=dt1.merge(dt2,on='QuestionID')
	dtf=dtf.merge(dt3,on='QuestionID')
	dtf=dtf.merge(dt4,on='QuestionID')

	print dtf.dtypes
	dtf.to_csv(os.path.join(os.path.dirname(__file__), 'data','feature.csv'), index=False)




	
	'''
	dta.NUM_TOKENS_IN_ANSWER.hist()
	plt.title('Histogram of NUM_TOKENS_IN_ANSWER')
	plt.xlabel('Token Count')
	plt.ylabel('Frequency')
	plt.show()
	
	
	
	

	dta['Appropriate'] = (dta.RATING > 0.0).astype(int)

	y, X = dmatrices('Appropriate ~ NUM_TOKENS_IN_ANSWER + NUM_TOKENS_IN_SENTENCE + NUM_RAW_TOKENS_MATCHING_IN_OUT+\
				PERCENT_TOKENS_IN_ANSWER+PERCENT_RAW_TOKENS_MATCHING_IN_OUT', dta, return_type="dataframe")
	y= np.ravel(y)
	
	model =  LogisticRegression(solver='lbfgs',max_iter=500,penalty='l2')
	model = model.fit(X, y)
	print model.score(X,y)

	# examine the coefficients
	print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
	'''

	

