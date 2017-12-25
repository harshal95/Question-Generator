import pandas as pd
#import matplotlib.pylab as pl
#import numpy as np
import os
#import ast
import LexHelp as lh
import SynHelp as sh
import nltk
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
    syntax_list=[]
    syndic={}
    ct=0
    for Sid,S,ans,ques,Qid,Jdg,Jid in zip(data.SentenceID,data.Sentence,data.Answer,data.Question,data.QuestionID,data.Judgment,data.JudgeId):
        
        if Qid=='None':
            continue

        if i==0:
             print 'on Question ',str(ct)
            # syndic['POS_1-GRAM_AFTER_ANSWER_CC'] = sh.pos1After(S,ans,"CC")	#binary	"The first POS tag following the answer span is CC"
             pos=sh.pos1After(S,ans)   
             syndic['QuestionID']=Qid         
             syndic['POS_1-GRAM_AFTER_ANSWER_CC'] = pos == "CC"	#binary	"The first POS tag following the answer span is CC"
             syndic['POS_1-GRAM_AFTER_ANSWER_CD'] = pos == "CD"	#binary	"The first POS tag following the answer span is CD"
             syndic['POS_1-GRAM_AFTER_ANSWER_DT'] = pos == "DT"	#binary	"The first POS tag following the answer span is DT"
             syndic['POS_1-GRAM_AFTER_ANSWER_EX'] = pos == "EX"	#binary	"The first POS tag following the answer span is EX"
             syndic['POS_1-GRAM_AFTER_ANSWER_IN'] = pos == "IN"	#binary	"The first POS tag following the answer span is IN"
             syndic['POS_1-GRAM_AFTER_ANSWER_JJ'] = pos == "JJ"	#binary	"The first POS tag following the answer span is JJ"
             syndic['POS_1-GRAM_AFTER_ANSWER_JJR'] =pos == "JJR"	#binary	"The first POS tag following the answer span is JJR"
             syndic['POS_1-GRAM_AFTER_ANSWER_JJS'] =pos == "JJS"	#binary	"The first POS tag following the answer span is JJS"
             syndic['POS_1-GRAM_AFTER_ANSWER_MD'] = pos == "MD"	#binary	"The first POS tag following the answer span is MD"
             syndic['POS_1-GRAM_AFTER_ANSWER_NN'] = pos == "NN"	#binary	"The first POS tag following the answer span is NN"
             syndic['POS_1-GRAM_AFTER_ANSWER_NNP'] =pos == "NNP"	#binary	"The first POS tag following the answer span is NNP"
             syndic['POS_1-GRAM_AFTER_ANSWER_NNPS']= pos == "NNPS"	#binary	"The first POS tag following the answer span is NNPS"
             syndic['POS_1-GRAM_AFTER_ANSWER_NNS'] =pos == "NNS" 	#binary	"The first POS tag following the answer span is NNS"
             syndic['POS_1-GRAM_AFTER_ANSWER_POS'] =pos == "POS" 	#binary	"The first POS tag following the answer span is POS"
             syndic['POS_1-GRAM_AFTER_ANSWER_PRP'] =pos == "PRP" 	#binary	"The first POS tag following the answer span is PRP"                 syndic['POS_1-GRAM_AFTER_ANSWER_PRPS'] = sh.pos1After(S,ans,"PRP$")	#binary	"The first POS tag following the answer span is PRP$"
             syndic['POS_1-GRAM_AFTER_ANSWER_RB'] = pos == "RB"	#binary	"The first POS tag following the answer span is RB"
             syndic['POS_1-GRAM_AFTER_ANSWER_RBR'] =pos == "RBR" 	#binary	"The first POS tag following the answer span is RBR"
             syndic['POS_1-GRAM_AFTER_ANSWER_RBS'] =pos == "RBS" 	#binary	"The first POS tag following the answer span is RBS"
             syndic['POS_1-GRAM_AFTER_ANSWER_RP'] = pos == "RP"	#binary	"The first POS tag following the answer span is RP"
             syndic['POS_1-GRAM_AFTER_ANSWER_TO'] = pos == "TO"	#binary	"The first POS tag following the answer span is TO"
             syndic['POS_1-GRAM_AFTER_ANSWER_VB'] = pos == "VB"
             	#binary	"The first POS tag following the answer span is VB"
             syndic['POS_1-GRAM_AFTER_ANSWER_VBD'] =pos == "VBD" 	#binary	"The first POS tag following the answer span is VBD"
             syndic['POS_1-GRAM_AFTER_ANSWER_VBG'] =pos == "VBG" 	#binary	"The first POS tag following the answer span is VBG"
             syndic['POS_1-GRAM_AFTER_ANSWER_VBN'] =pos == "VBN" 	#binary	"The first POS tag following the answer span is VBN"
             syndic['POS_1-GRAM_AFTER_ANSWER_VBP'] =pos == "VBP" 	#binary	"The first POS tag following the answer span is VBP"
             syndic['POS_1-GRAM_AFTER_ANSWER_VBZ'] =pos == "VBZ" 	#binary	"The first POS tag following the answer span is VBZ"
             syndic['POS_1-GRAM_AFTER_ANSWER_WDT'] =pos == "WDT" 	#binary	"The first POS tag following the answer span is WDT"
             syndic['POS_1-GRAM_AFTER_ANSWER_WP'] = pos == "WP"	#binary	"The first POS tag following the answer span is WP"
             syndic['POS_1-GRAM_AFTER_ANSWER_WRB'] =pos == "WRB" 	#binary	"The first POS tag following the answer span is WRB"
             pos=sh.pos1Before(S,ans)
             syndic['POS_1-GRAM_BEFORE_ANSWER_CC'] =pos == "CC"	#binary	"The first POS tag following the answer span is CC"
             syndic['POS_1-GRAM_BEFORE_ANSWER_CD'] =pos == "CD"	#binary	"The first POS tag following the answer span is CD"
             syndic['POS_1-GRAM_BEFORE_ANSWER_DT'] =pos == "DT"	#binary	"The first POS tag following the answer span is DT"
             syndic['POS_1-GRAM_BEFORE_ANSWER_EX'] =pos == "EX"	#binary	"The first POS tag following the answer span is EX"
             syndic['POS_1-GRAM_BEFORE_ANSWER_IN'] =pos == "IN"	#binary	"The first POS tag following the answer span is IN"
             syndic['POS_1-GRAM_BEFORE_ANSWER_JJ'] =pos == "JJ"	#binary	"The first POS tag following the answer span is JJ"
             syndic['POS_1-GRAM_BEFORE_ANSWER_JJR'] =pos == "JJR"	#binary	"The first POS tag following the answer span is JJR"
             syndic['POS_1-GRAM_BEFORE_ANSWER_JJS'] =pos == "JJS"	#binary	"The first POS tag following the answer span is JJS"
             syndic['POS_1-GRAM_BEFORE_ANSWER_MD'] = pos == "MD"	#binary	"The first POS tag following the answer span is MD"
             syndic['POS_1-GRAM_BEFORE_ANSWER_NN'] = pos == "NN"	#binary	"The first POS tag following the answer span is NN"
             syndic['POS_1-GRAM_BEFORE_ANSWER_NNP'] =pos == "NNP"	#binary	"The first POS tag following the answer span is NNP"
             syndic['POS_1-GRAM_BEFORE_ANSWER_NNPS'] =pos == "NNPS" 	#binary	"The first POS tag following the answer span is NNPS"
             syndic['POS_1-GRAM_BEFORE_ANSWER_NNS'] = pos == "NNS"	#binary	"The first POS tag following the answer span is NNS"
             syndic['POS_1-GRAM_BEFORE_ANSWER_POS'] = pos == "POS"	#binary	"The first POS tag following the answer span is POS"
             syndic['POS_1-GRAM_BEFORE_ANSWER_PRP'] = pos == "PRP"	#binary	"The first POS tag following the answer span is PRP"                 syndic['POS_1-GRAM_AFTER_ANSWER_PRPS'] = sh.pos1After(S,ans,"PRP$")	#binary	"The first POS tag following the answer span is PRP$"
             syndic['POS_1-GRAM_BEFORE_ANSWER_RB'] = pos == "RB"	#binary	"The first POS tag following the answer span is RB"
             syndic['POS_1-GRAM_BEFORE_ANSWER_RBR'] = pos == "RBR"	#binary	"The first POS tag following the answer span is RBR"
             syndic['POS_1-GRAM_BEFORE_ANSWER_RBS'] =pos == "RBS"	#binary	"The first POS tag following the answer span is RBS"
             syndic['POS_1-GRAM_BEFORE_ANSWER_RP'] = pos == "RP"	#binary	"The first POS tag following the answer span is RP"
             syndic['POS_1-GRAM_BEFORE_ANSWER_TO'] = pos == "TO"	#binary	"The first POS tag following the answer span is TO"
             syndic['POS_1-GRAM_BEFORE_ANSWER_VB'] = pos == "VB"	#binary	"The first POS tag following the answer span is VB"
             syndic['POS_1-GRAM_BEFORE_ANSWER_VBD'] =pos == "VBD"	#binary	"The first POS tag following the answer span is VBD"
             syndic['POS_1-GRAM_BEFORE_ANSWER_VBG'] = pos == "VBG"	#binary	"The first POS tag following the answer span is VBG"
             syndic['POS_1-GRAM_BEFORE_ANSWER_VBN'] = pos == "VBN"	#binary	"The first POS tag following the answer span is VBN"
             syndic['POS_1-GRAM_BEFORE_ANSWER_VBP'] = pos == "VBP"	#binary	"The first POS tag following the answer span is VBP"
             syndic['POS_1-GRAM_BEFORE_ANSWER_VBZ'] = pos == "VBZ"	#binary	"The first POS tag following the answer span is VBZ"
             syndic['POS_1-GRAM_BEFORE_ANSWER_WDT'] = pos == "WDT"	#binary	"The first POS tag following the answer span is WDT"
             syndic['POS_1-GRAM_BEFORE_ANSWER_WP'] = pos == "WP"	#binary	"The first POS tag following the answer span is WP"
             syndic['POS_1-GRAM_BEFORE_ANSWER_WRB'] = pos == "WRB"	#binary	"The first POS tag following the answer span is WRB"
             new_dic=sh.pos1In(ans)
             syndic['POS_1-GRAM_IN_ANSWER_COUNT_CC']=new_dic.get('CC',0)#"The number of tokens with POS tag CC in the answer"
  	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_CD']=new_dic.get('CD',0)		#"The number of tokens with POS tag CD in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_DT']=new_dic.get('DT',0)		#"The number of tokens with POS tag DT in the answer"
             syndic['POS_1-GRAM_IN_ANSWER_COUNT_EX']=new_dic.get('EX',0)		#"The number of tokens with POS tag EX in the answer"
             syndic['POS_1-GRAM_IN_ANSWER_COUNT_IN']=new_dic.get('IN',0)		#"The number of tokens with POS tag IN in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_JJ']=new_dic.get('JJ',0)	#"The number of tokens with POS tag JJ in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_JJR']=new_dic.get('JJR',0)	#"The number of tokens with POS tag JJR in the answer"
             syndic['POS_1-GRAM_IN_ANSWER_COUNT_JJS']=new_dic.get('JJS',0)		#"The number of tokens with POS tag JJS in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_MD']=new_dic.get('MD',0)	#"The number of tokens with POS tag MD in the answer"
             syndic['POS_1-GRAM_IN_ANSWER_COUNT_NN']=new_dic.get('NN',0)		#"The number of tokens with POS tag NN in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_NNP']=new_dic.get('NNP',0)	#"The number of tokens with POS tag NNP in the answer"
             syndic['POS_1-GRAM_IN_ANSWER_COUNT_NNPS']=	new_dic.get('NNPS',0)	#"The number of tokens with POS tag NNPS in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_NNS']=new_dic.get('NNS',0)	#"The number of tokens with POS tag NNS in the answer"
             syndic['POS_1-GRAM_IN_ANSWER_COUNT_POS']=new_dic.get('POS',0)	#"The number of tokens with POS tag POS in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_PRP']=new_dic.get('PRP',0)	#"The number of tokens with POS tag PRP in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_PRP$']=new_dic.get('PRP$',0)		#"The number of tokens with POS tag PRP$ in the answer"
             syndic['POS_1-GRAM_IN_ANSWER_COUNT_RB']=new_dic.get('RB',0)		#"The number of tokens with POS tag RB in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_RBR']=new_dic.get('RBR',0)	#"The number of tokens with POS tag RBR in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_RBS']=new_dic.get('RBS',0)	#"The number of tokens with POS tag RBS in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_RP']=new_dic.get('RP',0)	#"The number of tokens with POS tag RP in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_TO']=new_dic.get('TO',0)	#"The number of tokens with POS tag TO in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_VB']=new_dic.get('VB',0)	#"The number of tokens with POS tag VB in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_VBD']=new_dic.get('VBD',0)	#"The number of tokens with POS tag VBD in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_VBG']=new_dic.get('VBG',0)	#"The number of tokens with POS tag VBG in the answer"
             syndic['POS_1-GRAM_IN_ANSWER_COUNT_VBN']=new_dic.get('VBN',0)		#"The number of tokens with POS tag VBN in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_VBP']=new_dic.get('VBP',0)	#"The number of tokens with POS tag VBP in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_VBZ']=new_dic.get('VBZ',0)	#"The number of tokens with POS tag VBZ in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_WDT']=new_dic.get('WDT',0)	#"The number of tokens with POS tag WDT in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_WP']=new_dic.get('WP',0)#"The number of tokens with POS tag WP in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_WP$']=new_dic.get('WP$',0)	#"The number of tokens with POS tag WP$ in the answer"
	     syndic['POS_1-GRAM_IN_ANSWER_COUNT_WRB']=new_dic.get('WRB',0)	#"The number of tokens with POS tag WRB in the answer"
	     new_dic={}
          

             ct = ct + 1
        
        if i==3:
             syntax_list.append(syndic)
             syndic={}
             
        i=(i+1)%4
         
        
    dta=pd.DataFrame.from_dict(syntax_list, orient='columns')
    dta.to_csv(os.path.join(os.path.dirname(__file__), 'data','syntactic_feature.csv'), index=False)
    


