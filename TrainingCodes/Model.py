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

def numberofTokens(S):
	
	S=str(S)
	words=S.split()
	return len(words)

def allCapital(S): #all caps
	
	S=str(S)
	words=S.split()
	ct=0
	for x in words:
		if x==x.upper():
			ct+=1
	return 	ct

def startCapital(S):
	S=str(S)
	words=S.split()
	ct=0
	for x in words:
		if x[0]>='A' and x[0]<='Z':
			ct+=1
	return ct

def numberofinout(S,ans):

	S=str(S)
	ans=str(ans)
	words_S=S.split()
	words_ans=ans.split()
	ct=0
	for w in words_ans:
		if w in words_S:
			ct+=1

	return ct		 		

def getRating(Jdg):
	Jdg=str(Jdg)
	if Jdg=="Good":
		return 1.0
	elif Jdg=="Bad":
		return -1.0
	else:
		return 0.5
	
if __name__ == '__main__':

	dtf = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'Train.csv'))
	data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'MindTheGap-1.0-Data.tsv'), header=0, \
                    delimiter="\t", quoting=3)

	print "Data available for a sentence"
	print dtf.dtypes
	print 
	print "Data's Structure"
	print data.shape

	sid=data.SentenceID[0]
	
	'''
	for Sid,S,ans,ques,Jdg,Jid in zip(data.SentenceID,data.Sentence,data.Answer,data.Question,data.Judgment,data.JudgeId):
			if(sid==Sid):
					print ques
	'''
		
	i=0
	
	#Token Count Features
	token_ct_list=[]
	tokdic={}
	for Sid,S,ans,ques,Jdg,Jid,Qid in zip(data.SentenceID,data.Sentence,data.Answer,data.Question,data.Judgment,data.JudgeId,data.QuestionID):


		if Qid=='None':
			continue		
		if i==0:
			tokdic['NUM_TOKENS_IN_ANSWER']=numberofTokens(ans)
			tokdic['NUM_TOKENS_IN_SENTENCE']=numberofTokens(S)
			tokdic['NUM_RAW_TOKENS_MATCHING_IN_OUT']=numberofinout(S,ans)
			tokdic['PERCENT_TOKENS_IN_ANSWER']=float(tokdic['NUM_TOKENS_IN_ANSWER'])/tokdic['NUM_TOKENS_IN_SENTENCE']
			tokdic['PERCENT_RAW_TOKENS_MATCHING_IN_OUT']=float(tokdic['NUM_RAW_TOKENS_MATCHING_IN_OUT'])/tokdic['NUM_TOKENS_IN_ANSWER']
			tokdic['RATING']=getRating(Jdg)
		else:
			tokdic['RATING']+=getRating(Jdg)
		if i==3:	
			token_ct_list.append(tokdic)
			tokdic={}
			
		i=(i+1)%4
		
	print len(token_ct_list)	
	dta=pd.DataFrame.from_dict(token_ct_list, orient='columns', dtype=None)

	
	dta.NUM_TOKENS_IN_ANSWER.hist()
	plt.title('Histogram of NUM_TOKENS_IN_ANSWER')
	plt.xlabel('Token Count')
	plt.ylabel('Frequency')
	plt.show()
	
	dta['Appropriate'] = (dta.RATING >= 1.5).astype(int)

	y, X = dmatrices('Appropriate ~ NUM_TOKENS_IN_ANSWER + NUM_TOKENS_IN_SENTENCE +NUM_RAW_TOKENS_MATCHING_IN_OUT+\
				PERCENT_TOKENS_IN_ANSWER+PERCENT_RAW_TOKENS_MATCHING_IN_OUT', dta, return_type="dataframe")
	y= np.ravel(y)
	
	model =  LogisticRegression(solver='lbfgs',max_iter=500,penalty='l2')
	model = model.fit(X, y)
	print model.score(X,y)

	# examine the coefficients
	print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
	
	


