import Msrtool
import pandas as pd
import matplotlib.pylab as pl
import numpy as np
import os
import sys
if __name__ == '__main__':

	
	reload(sys)
	sys.setdefaultencoding('utf-8')
	data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'MindTheGap-1.0-Data.tsv'), header=0, \
                    delimiter="\t", quoting=3)
	
	analyzers = "Constituency_Tree,Semantic_Roles,POS_Tags,Named_Entities"
	Sentence_Dict={}
	
	for S,Sid in zip(data.Sentence,data.SentenceID):
		if Sid not in Sentence_Dict:
			Sentence_Dict[Sid]=S

	n=len(Sentence_Dict)
	
	i=0	
	dic_list=[]
	for key in Sentence_Dict:
		dic=Msrtool.getList(analyzers,Sentence_Dict[key],key)
		dic_list.append(dic)
		print 'On sentence',str(i+1),'of',str(n)
		i=i+1
		
				
	df=pd.DataFrame.from_dict(dic_list, orient='columns', dtype=None)

	print df.dtypes
	df.to_csv('Train.csv', index=False)
	print('Done')
