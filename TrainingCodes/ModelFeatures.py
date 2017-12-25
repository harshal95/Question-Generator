import pandas as pd
import matplotlib.pylab as pl
import numpy as np
import os
import ast
import LexHelp as lh
import nltk
import csv
from nltk.corpus import stopwords 


	
if __name__ == '__main__':

	dtf = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'Train.csv'))
	data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'MindTheGap-1.0-Data.tsv'), header=0, \
                    delimiter="\t", quoting=3)

	print "Data available for a sentence"
	print dtf.dtypes
	print 
	print "Data's Structure"
	print data.dtypes

		
	i=0

	#POS TAGS List
	posdic={}
	for Sid,POS in zip(dtf.SentenceID,dtf.POS_Tags):

			x=ast.literal_eval(POS)
			posdic[Sid]=x[0] 

	#print posdic		


	


	
	#Token Count Features
	ct=0
	token_ct_list=[]
	tokdic={}
	for Sid,S,ans,ques,Jdg,Qid in zip(data.SentenceID,data.Sentence,data.Answer,data.Question,data.Judgment,data.QuestionID):

		if Qid=='None':
			continue;
		
		if i==0:
			tokdic['QuestionID']=Qid
			ct+=1
			#tokdic['NUM_TOKENS_IN_ANSWER']=lh.numberofTokens(ans)
			#tokdic['NUM_TOKENS_IN_SENTENCE']=lh.numberofTokens(S)
			#tokdic['NUM_RAW_TOKENS_MATCHING_IN_OUT']=lh.numberofinout(S,ans)
			tokdic['PERCENT_TOKENS_IN_ANSWER']=float(lh.numberofTokens(ans))/lh.numberofTokens(S)
			tokdic['PERCENT_RAW_TOKENS_MATCHING_IN_OUT']=float(lh.numberofinout(S,ans))/lh.numberofTokens(ans)
			tokdic['RATING']=lh.getRating(Jdg)
		else:
			tokdic['RATING']+=lh.getRating(Jdg)
		if i==3:	
			token_ct_list.append(tokdic)	
			tokdic={}
			
		i=(i+1)%4

		
	print len(token_ct_list)
	
	dta=pd.DataFrame.from_dict(token_ct_list, orient='columns')
	dta.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'token_ct_features.csv'), index=False)

	
	i=0
    #Lexical Features
	lexical_list=[]
	lexdic={}
	ct=0
	for Sid,S,ans,ques,Qid,Jdg,Jid in zip(data.SentenceID,data.Sentence,data.Answer,data.Question,data.QuestionID,data.Judgment,data.JudgeId):
		
		if Qid=='None':
			continue

		#for a new sentence
		if i==0:
			anslen=lh.numberofTokens(ans)
			print 'on Question',str(ct)
			lexdic['QuestionID']=Qid
			lexdic['ANSWER_CAPITALIZED_WORD_DENSITY']=float(lh.startCapital(ans))/anslen #"Percentage of tokens in the answer that are all caps"
			lexdic['ANSWER_ABBREVIATION_WORD_DENSITY']=float(lh.allCapital(ans))/anslen
													#"Percentage of tokens in the answer that are abbreviations"
			lexdic['ANSWER_PRONOMINAL_DENSITY']=float(lh.pronounCount(ans))/anslen		#"Percentage of tokens in the answer that are pronouns"
			lexdic['ANSWER_STOPWORD_DENSITY']=float(lh.stopCount(ans))/anslen	#"Percentage of tokens in the answer that are stopwords"
			#lexdic['ANSWER_ENDS_WITH_QUANTIFIER']=False	#"Answer ends with a quantifier word (many, few, etc.)"
			#lexdic['ANSWER_STARTS_WITH_QUANTIFIER']=False	#"First word in the answer starts with a quantifier word"
			lexdic['ANSWER_QUANTIFIER_DENSITY']=0		#"Percentage of tokens in the answer that are quantifier words"
			temp=lh.startCapital(S)
			if temp!=0:
				lexdic['PERCENT_CAPITALIZED_WORDS_IN_ANSWER']=float(lh.startCapital(ans))/lh.startCapital(S)
	   		temp=lh.allCapital(S)
	   		if temp!=0:					        #"Percentage of capitalized words in the sentence that are in the answer"
				lexdic['PERCENT_ABBREVIATED_WORDS_IN_ANSWER']=float(lh.allCapital(ans))/lh.allCapital(S) #"Percentage of abbreviations in the sentence that are in the answer"
			temp=lh.pronounCount(S);
			if temp!=0:					
				lexdic['PERCENT_PRONOMINALS_IN_ANSWER']=float((lexdic['ANSWER_PRONOMINAL_DENSITY']*anslen))/temp
				#"Percentage of pronouns in the sentence that are in the answer"
			#lexdic['SENTENCE_STARTS_WITH_DISCOURSE_CONNECTIVE']=False #"Sentence starts with a discourse connective word"

			lexdic['RATING']=lh.getRating(Jdg)
			ct+=1
		else:
				lexdic['RATING']+=lh.getRating(Jdg)

		if i==3:	
				lexical_list.append(lexdic)
				lexdic={}
				
		i=(i+1)%4
		
		
		

	dta=pd.DataFrame.from_dict(lexical_list, orient='columns')
	dta.to_csv(os.path.join(os.path.dirname(__file__), 'data','lexical_feature.csv'), index=False)
	

	
	



