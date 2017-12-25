import pandas as pd
import matplotlib.pylab as pl
import numpy as np
import os
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from sklearn.cross_validation import cross_val_score


if __name__ == '__main__':

	
	
	dta = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'feature.csv'))
	dta1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'TestData.csv'))
	
	lis1=[]
	lis1=dta.columns
	#print dta.columns
	#print dta.dtypes

	
	
	dta['RATING'] = (dta.RATING >= 1.5).astype(int)


	'''
	y, X = dmatrices('RATING~ NUM_TOKENS_IN_ANSWER + NUM_TOKENS_IN_SENTENCE + NUM_RAW_TOKENS_MATCHING_IN_OUT+\
	
			PERCENT_TOKENS_IN_ANSWER+PERCENT_RAW_TOKENS_MATCHING_IN_OUT', dta, return_type="dataframe")
	'''
	
	f = 'RATING ~ POS_1_GRAM_AFTER_ANSWER_CC + POS_1_GRAM_AFTER_ANSWER_CD + POS_1_GRAM_AFTER_ANSWER_DT + POS_1_GRAM_AFTER_ANSWER_EX + POS_1_GRAM_AFTER_ANSWER_IN + POS_1_GRAM_AFTER_ANSWER_JJ + POS_1_GRAM_AFTER_ANSWER_JJR + POS_1_GRAM_AFTER_ANSWER_JJS + POS_1_GRAM_AFTER_ANSWER_MD + POS_1_GRAM_AFTER_ANSWER_NN + POS_1_GRAM_AFTER_ANSWER_NNP + POS_1_GRAM_AFTER_ANSWER_NNPS + POS_1_GRAM_AFTER_ANSWER_NNS + POS_1_GRAM_AFTER_ANSWER_POS + POS_1_GRAM_AFTER_ANSWER_PRP + POS_1_GRAM_AFTER_ANSWER_RB + POS_1_GRAM_AFTER_ANSWER_RBR + POS_1_GRAM_AFTER_ANSWER_RBS + POS_1_GRAM_AFTER_ANSWER_RP + POS_1_GRAM_AFTER_ANSWER_TO + POS_1_GRAM_AFTER_ANSWER_VB + POS_1_GRAM_AFTER_ANSWER_VBD + POS_1_GRAM_AFTER_ANSWER_VBG + POS_1_GRAM_AFTER_ANSWER_VBN + POS_1_GRAM_AFTER_ANSWER_VBP + POS_1_GRAM_AFTER_ANSWER_VBZ + POS_1_GRAM_AFTER_ANSWER_WDT + POS_1_GRAM_AFTER_ANSWER_WP + POS_1_GRAM_AFTER_ANSWER_WRB + POS_1_GRAM_BEFORE_ANSWER_CC + POS_1_GRAM_BEFORE_ANSWER_CD + POS_1_GRAM_BEFORE_ANSWER_DT + POS_1_GRAM_BEFORE_ANSWER_EX + POS_1_GRAM_BEFORE_ANSWER_IN + POS_1_GRAM_BEFORE_ANSWER_JJ + POS_1_GRAM_BEFORE_ANSWER_JJR + POS_1_GRAM_BEFORE_ANSWER_JJS + POS_1_GRAM_BEFORE_ANSWER_MD + POS_1_GRAM_BEFORE_ANSWER_NN + POS_1_GRAM_BEFORE_ANSWER_NNP + POS_1_GRAM_BEFORE_ANSWER_NNPS + POS_1_GRAM_BEFORE_ANSWER_NNS + POS_1_GRAM_BEFORE_ANSWER_POS + POS_1_GRAM_BEFORE_ANSWER_PRP + POS_1_GRAM_BEFORE_ANSWER_RB + POS_1_GRAM_BEFORE_ANSWER_RBR + POS_1_GRAM_BEFORE_ANSWER_RBS + POS_1_GRAM_BEFORE_ANSWER_RP + POS_1_GRAM_BEFORE_ANSWER_TO + POS_1_GRAM_BEFORE_ANSWER_VB + POS_1_GRAM_BEFORE_ANSWER_VBD + POS_1_GRAM_BEFORE_ANSWER_VBG + POS_1_GRAM_BEFORE_ANSWER_VBN + POS_1_GRAM_BEFORE_ANSWER_VBP + POS_1_GRAM_BEFORE_ANSWER_VBZ + POS_1_GRAM_BEFORE_ANSWER_WDT + POS_1_GRAM_BEFORE_ANSWER_WP + POS_1_GRAM_BEFORE_ANSWER_WRB + POS_1_GRAM_IN_ANSWER_COUNT_CC + POS_1_GRAM_IN_ANSWER_COUNT_CD + POS_1_GRAM_IN_ANSWER_COUNT_DT + POS_1_GRAM_IN_ANSWER_COUNT_EX + POS_1_GRAM_IN_ANSWER_COUNT_IN + POS_1_GRAM_IN_ANSWER_COUNT_JJ + POS_1_GRAM_IN_ANSWER_COUNT_JJR + POS_1_GRAM_IN_ANSWER_COUNT_JJS + POS_1_GRAM_IN_ANSWER_COUNT_MD + POS_1_GRAM_IN_ANSWER_COUNT_NN + POS_1_GRAM_IN_ANSWER_COUNT_NNP + POS_1_GRAM_IN_ANSWER_COUNT_NNPS + POS_1_GRAM_IN_ANSWER_COUNT_NNS + POS_1_GRAM_IN_ANSWER_COUNT_POS + POS_1_GRAM_IN_ANSWER_COUNT_PRP + POS_1_GRAM_IN_ANSWER_COUNT_RB + POS_1_GRAM_IN_ANSWER_COUNT_RBR + POS_1_GRAM_IN_ANSWER_COUNT_RBS + POS_1_GRAM_IN_ANSWER_COUNT_RP + POS_1_GRAM_IN_ANSWER_COUNT_TO + POS_1_GRAM_IN_ANSWER_COUNT_VB + POS_1_GRAM_IN_ANSWER_COUNT_VBD + POS_1_GRAM_IN_ANSWER_COUNT_VBG + POS_1_GRAM_IN_ANSWER_COUNT_VBN + POS_1_GRAM_IN_ANSWER_COUNT_VBP + POS_1_GRAM_IN_ANSWER_COUNT_VBZ + POS_1_GRAM_IN_ANSWER_COUNT_WDT + POS_1_GRAM_IN_ANSWER_COUNT_WP + POS_1_GRAM_IN_ANSWER_COUNT_WRB + ANSWER_CONTAINS_SRL_A0+ PERCENT_RAW_TOKENS_MATCHING_IN_OUT + PERCENT_TOKENS_IN_ANSWER+ANSWER_ABBREVIATION_WORD_DENSITY + ANSWER_CAPITALIZED_WORD_DENSITY + ANSWER_PRONOMINAL_DENSITY + ANSWER_QUANTIFIER_DENSITY + ANSWER_STOPWORD_DENSITY+ ANSWER_CONTAINS_SRL_A1 + ANSWER_CONTAINS_SRL_A2 + ANSWER_CONTAINS_SRL_A3+ANSWER_CONTAINS_SRL_AM_ADV + ANSWER_CONTAINS_SRL_AM_DIS + ANSWER_CONTAINS_SRL_AM_EXT + ANSWER_CONTAINS_SRL_AM_LOC+ANSWER_CONTAINS_SRL_AM_MNR + ANSWER_CONTAINS_SRL_AM_MOD+ANSWER_CONTAINS_SRL_AM_PNC + ANSWER_CONTAINS_SRL_AM_REC + ANSWER_CONTAINS_SRL_AM_TMP + ANSWER_CONTAINS_SRL_C_A0 + ANSWER_CONTAINS_SRL_C_A1 + ANSWER_CONTAINS_SRL_R_A0+ANSWER_CONTAINS_SRL_R_A1 + ANSWER_CONTAINS_SRL_R_A2 + ANSWER_CONTAINS_SRL_R_AM_LOC + ANSWER_CONTAINS_SRL_R_AM_MNR+ANSWER_CONTAINS_SRL_R_AM_TMP + ANSWER_CONTAINS_SRL_predicate + ANSWER_COVERED_BY_SRL_A0 + ANSWER_COVERED_BY_SRL_A1 + ANSWER_COVERED_BY_SRL_A2+ANSWER_COVERED_BY_SRL_A3 + ANSWER_COVERED_BY_SRL_A4 + ANSWER_COVERED_BY_SRL_AM_ADV + ANSWER_COVERED_BY_SRL_AM_CAU+ANSWER_COVERED_BY_SRL_AM_DIR + ANSWER_COVERED_BY_SRL_AM_DIS + ANSWER_COVERED_BY_SRL_AM_LOC + ANSWER_COVERED_BY_SRL_AM_MNR+ANSWER_COVERED_BY_SRL_AM_PNC + ANSWER_COVERED_BY_SRL_AM_REC + ANSWER_COVERED_BY_SRL_AM_TMP + ANSWER_COVERED_BY_SRL_C_A0+ANSWER_COVERED_BY_SRL_C_A1+ANSWER_COVERED_BY_SRL_predicate' 
	y,X= dmatrices(f,dta,return_type="dataframe")
	#print X.shape
	
	
		
	y= np.ravel(y)
	

	model =  LogisticRegression(solver='lbfgs',max_iter=100,penalty='l2')

	
	model = model.fit(X,y)

	#print model.score(X,y)

	
	#print X.shape
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	model2 = LogisticRegression(solver='liblinear',max_iter=100,penalty='l2')

	#print y_train
	#print y_train
	
	
	result=model2.fit(X_train, y_train)
	print result

	predicted = model2.predict(X_test)
	#print X_test.columns
	
	#print predicted
	probs = model2.predict_proba(X_test)
	#print probs
	
	print metrics.accuracy_score(y_test, predicted)
	
	actual=y_test
	predictions=predicted
	#for p,p1 in zip(probs,predicted):
		#print p[1],p1
	for i in range(0,len(predictions)):
		if probs[i][1]>=0.56:
			predictions[i]=1
		else:
			predictions[i]=0
			
		
	false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	plt.title('Receiver Operating Characteristic')
	plt.plot(false_positive_rate, true_positive_rate, 'b',
	label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.05])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()


	#print metrics.roc_auc_score(y_test, probs[:, 1])
	
	# examine the coefficients
	#print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
	
	

	

