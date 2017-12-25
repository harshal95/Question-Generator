import pandas as pd
import matplotlib.pylab as pl
import numpy as np
import os
import nltk

if __name__ == '__main__':

	dtf = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'Train.csv'))
	data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'MindTheGap-1.0-Data.tsv'), header=0, \
                    delimiter="\t", quoting=3)

	print dtf.dtypes
	
	posdic_list=[]
	i=0
	for S,Sid in zip(dtf.Sentence,dtf.SentenceID):
		print 'on sentence',str(i)
		posdic={}
		posdic['SentenceID']=Sid
		posdic['POS']=nltk.pos_tag(nltk.word_tokenize(S))
		i=i+1
		posdic_list.append(posdic)

	print posdic_list
	
	
	dfpos=pd.DataFrame.from_dict(posdic_list, orient='columns', dtype=None)

	print dfpos.dtypes
	dfpos.to_csv('POS.csv', index=False)
	
		

