import pandas as pd
import ast
import nltk
from nltk.tree import *
import re
import pkgutil
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import re
import pkgutil
import requests
from bllipparser import RerankingParser
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
import SynHelp as sh
import LexHelp as lh
import matplotlib.pylab as pl
import numpy as np
import os
import matplotlib.pyplot as plt
from patsy import dmatrices
from patsy import dmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import operator
from termcolor import colored
import pickle


#contains the tokenized representation of a sentence
tokens=[]

def getPositionTuple(S,ans):

    lis=[]
    offset=len(ans)-1

    S=' '.join(S) 
    ans=' '.join(ans)

    pos=S.find(ans)
    if(pos!=-1):
        st=len(S[:pos].split(' '))
        return(st-1,st-1+offset)
       




def splitkeepsep(s, sep):
    return reduce(lambda acc, elem: acc[:-1] + [acc[-1] + elem] if elem == sep else acc + [elem], re.split("(%s)" % re.escape(sep), s), [])

LANGUAGE =  "english"
SENTENCES_COUNT = 1

ans_list=[]

def getRoles(analyzers,sentence):
    language = "en"
    request = "http://msrsplat.cloudapp.net/SplatServiceJson.svc/Analyze?language="+language+"&analyzers="+analyzers+"&appId=358378CB-9C65-43AD-A365-FF7D6AD85620&json=x&input="

    sentence_request=request+sentence

    r=requests.get(sentence_request);
    results=r.json()
    dic={}
    dic['Sentence']=sentence
    for item in results:
        dic[item['Key']]=item['Value']

    return dic



def getList(S,srole):

    x=srole
       
    x= ast.literal_eval(x)
    slrs=x[0]

    #the position of the word from the begining
    tokens=nltk.word_tokenize(S)
    #print tokens
    Sn=' '.join(tokens)

    wordlist=[]
    for slr in slrs:
        slr=slr.split('[')[1:]
        
        for tag in slr:
            
            tuples=re.findall(r'([A-Z0-9\-]+)\=([0-9]+\-[0-9]+)',tag)
            for sl in tuples:
                (lab,rng)=sl
                rng=re.findall(r'([0-9]+)\-([0-9]+)',rng)
                (st,end)=rng[0]
                st=int(st)
                end=int(end)
                wordlist.append((st,end))

    #print wordlist
        
    
    treeS=rrp.simple_parse(Sn)
    
    tree =Tree.fromstring(treeS)
    t=tree
    height=t.height()

    height=min(height,5)
    qlist=[]
    for h in range(2,height):
        #print 'level:',str(h)
        for s in t.subtrees(lambda t: t.height() == h):
            try:
                s.label()
            except AttributeError:
                continue
            else:
                if s.label()=='NP' or s.label()=='VP':
                    ans1=s.leaves()
                    ans=[]
                    for item in ans1:
                        if item.lower() not in stopwords.words("english"):
                            ans.append(item)
                    try:
                        (ab,ae)=getPositionTuple(tokens,ans)
                        #print (ab,ae)
                        #find whether the list returned is found in sem wordlist
                        for (st,end) in wordlist:
                            if ab>=st and ae<=end :
                                qlist.append((ab,ae))
                    except:
                        continue


    return set(qlist)

def initialize(semdic):
    
    semdic['ANSWER_COVERED_BY_SRL_A0']=False        #"Does a span labeled A0 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_A1']=False          #"Does a span labeled A1 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_A2']=False            #"Does a span labeled A2 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_A3']=False        #"Does a span labeled A3 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_A4']=False            #"Does a span labeled A4 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM_ADV']=False        #"Does a span labeled AM_ADV cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM_CAU']=False        #"Does a span labeled AM_CAU cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM_DIR']=False        #"Does a span labeled AM_DIR cover the answer"  
    semdic['ANSWER_COVERED_BY_SRL_AM_DIS']=False        #"Does a span labeled AM_DIS cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM_LOC']=False        #"Does a span labeled AM_LOC cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM_MNR']=False        #"Does a span labeled AM_MNR cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM_PNC']=False        #"Does a span labeled AM_PNC cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM_REC']=False        #"Does a span labeled AM_REC cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM_TMP']=False        #"Does a span labeled AM_TMP cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_C_A0']=False          #"Does a span labeled C_A0 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_C_A1']=False          #"Does a span labeled C_A1 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_predicate']=False     #"Does a span labeled predicate cover the answer"
    semdic['ANSWER_CONTAINS_SRL_A0']=False              #"Is an SRL span labeled A0 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_A1']=False             #"Is an SRL span labeled A1 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_A2']=False              #"Is an SRL span labeled A2 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_A3']=False              #"Is an SRL span labeled A3 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM_ADV']=False          #"Is an SRL span labeled AM_ADV contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM_DIS']=False          #"Is an SRL span labeled AM_DIS contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM_EXT']=False          #"Is an SRL span labeled AM_EXT contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM_LOC']=False          #"Is an SRL span labeled AM_LOC contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM_MNR']=False          #"Is an SRL span labeled AM_MNR contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM_MOD']=False          #"Is an SRL span labeled AM_MOD contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM_PNC']=False          #"Is an SRL span labeled AM_PNC contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM_REC']=False         #"Is an SRL span labeled AM_REC contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM_TMP']=False          #"Is an SRL span labeled AM_TMP contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_C_A0']=False            #"Is an SRL span labeled C_A0 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_C_A1']=False           #"Is an SRL span labeled C_A1 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R_A0']=False           #"Is an SRL span labeled R_A0 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R_A1']=False          #"Is an SRL span labeled R_A1 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R_A2']=False             #"Is an SRL span labeled R_A2 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R_AM_LOC']=False        #"Is an SRL span labeled R_AM_LOC contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R_AM_MNR']=False        #"Is an SRL span labeled R_AM_MNR contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R_AM_TMP']=False        #"Is an SRL span labeled R_AM_TMP contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_predicate']=False       #"Is an SRL span labeled predicate contained within the span of the answer"
    
    return semdic

def sem_cover_contain(S,ans,srole,semdic):

    x=srole
    x= ast.literal_eval(x)
    slrs=x[0]

    #the position of the word from the begining
    Sn=' '.join(nltk.word_tokenize(S))
    ans=' '.join(nltk.word_tokenize(ans))
    ind1= Sn.find(ans)

    ans_st=len(S[0:ind1].split())
    

    inter=(ans_st,ans_st+len(nltk.word_tokenize(ans))-1)

    (i1,i2)=inter

    
    
    for slr in slrs:
        slr=slr.split('[')[1:]
        
        for tag in slr:
            #print tag
            tuples=re.findall(r'([A-Z0-9\-]+)\=([0-9]+\-[0-9]+)',tag)
            for sl in tuples:
                (lab,rng)=sl
                rng=re.findall(r'([0-9]+)\-([0-9]+)',rng)
                (st,end)=rng[0]
                st=int(st)
                end=int(end)
                
                
                
                #completely within
                if(st<=i1 and i2<=end and lab[0]!='R' and lab!='AM-MOD' and lab!='AM-NEG'):
                    semdic['ANSWER_COVERED_BY_SRL_'+lab]=True
                    
                    
                #Matches whole
                if(i1<=st and end<=i2):
                    semdic['ANSWER_CONTAINS_SRL_'+lab]=True
                    
                    
    return semdic
                    
                

 
rrp=0
import os
import sys
if __name__ == '__main__':



    SENTENCES_COUNT=5

    try:
        if len(sys.argv)>1:
            SENTENCES_COUNT=int(sys.argv[1])
    except:
        SENTENCES_COUNT=1


    parser = PlaintextParser.from_file(os.path.join(os.path.dirname(__file__),'Passage','passage.txt'), Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    i=0  
    
    semlist=[]
    
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        semrole={}
        print 'Getting SemRoles of Sentence ',str(i),' in progress...'
        semrole=getRoles("Semantic_Roles",str(sentence))
        semrole['Sentence_ID']=i
	
        semlist.append(semrole);
        semrole={}    
        i=i+1
    

    df=pd.DataFrame.from_dict(semlist, orient='columns', dtype=None)

        
    print df.dtypes
    df.to_csv('semroles.csv', index=False)

   
    
    df=pd.read_csv('semroles.csv')

    rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
   

    i=0
    semdic_list=[]
    semdic={}
    semdic=initialize(semdic)
    syntax_list=[]
    syndic={}
    token_ct_list=[]
    tokdic={}
    lexical_list=[]
    lexdic={}
    anslist=[]
    dta = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'feature.csv'))
    dta1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'TestData.csv'))
    dta['RATING'] = (dta.RATING >= 1.5).astype(int)
    f = 'RATING ~ POS_1_GRAM_AFTER_ANSWER_CC + POS_1_GRAM_AFTER_ANSWER_CD + POS_1_GRAM_AFTER_ANSWER_DT + POS_1_GRAM_AFTER_ANSWER_EX + POS_1_GRAM_AFTER_ANSWER_IN + POS_1_GRAM_AFTER_ANSWER_JJ + POS_1_GRAM_AFTER_ANSWER_JJR + POS_1_GRAM_AFTER_ANSWER_JJS + POS_1_GRAM_AFTER_ANSWER_MD + POS_1_GRAM_AFTER_ANSWER_NN + POS_1_GRAM_AFTER_ANSWER_NNP + POS_1_GRAM_AFTER_ANSWER_NNPS + POS_1_GRAM_AFTER_ANSWER_NNS + POS_1_GRAM_AFTER_ANSWER_POS + POS_1_GRAM_AFTER_ANSWER_PRP + POS_1_GRAM_AFTER_ANSWER_RB + POS_1_GRAM_AFTER_ANSWER_RBR + POS_1_GRAM_AFTER_ANSWER_RBS + POS_1_GRAM_AFTER_ANSWER_RP + POS_1_GRAM_AFTER_ANSWER_TO + POS_1_GRAM_AFTER_ANSWER_VB + POS_1_GRAM_AFTER_ANSWER_VBD + POS_1_GRAM_AFTER_ANSWER_VBG + POS_1_GRAM_AFTER_ANSWER_VBN + POS_1_GRAM_AFTER_ANSWER_VBP + POS_1_GRAM_AFTER_ANSWER_VBZ + POS_1_GRAM_AFTER_ANSWER_WDT + POS_1_GRAM_AFTER_ANSWER_WP + POS_1_GRAM_AFTER_ANSWER_WRB + POS_1_GRAM_BEFORE_ANSWER_CC + POS_1_GRAM_BEFORE_ANSWER_CD + POS_1_GRAM_BEFORE_ANSWER_DT + POS_1_GRAM_BEFORE_ANSWER_EX + POS_1_GRAM_BEFORE_ANSWER_IN + POS_1_GRAM_BEFORE_ANSWER_JJ + POS_1_GRAM_BEFORE_ANSWER_JJR + POS_1_GRAM_BEFORE_ANSWER_JJS + POS_1_GRAM_BEFORE_ANSWER_MD + POS_1_GRAM_BEFORE_ANSWER_NN + POS_1_GRAM_BEFORE_ANSWER_NNP + POS_1_GRAM_BEFORE_ANSWER_NNPS + POS_1_GRAM_BEFORE_ANSWER_NNS + POS_1_GRAM_BEFORE_ANSWER_POS + POS_1_GRAM_BEFORE_ANSWER_PRP + POS_1_GRAM_BEFORE_ANSWER_RB + POS_1_GRAM_BEFORE_ANSWER_RBR + POS_1_GRAM_BEFORE_ANSWER_RBS + POS_1_GRAM_BEFORE_ANSWER_RP + POS_1_GRAM_BEFORE_ANSWER_TO + POS_1_GRAM_BEFORE_ANSWER_VB + POS_1_GRAM_BEFORE_ANSWER_VBD + POS_1_GRAM_BEFORE_ANSWER_VBG + POS_1_GRAM_BEFORE_ANSWER_VBN + POS_1_GRAM_BEFORE_ANSWER_VBP + POS_1_GRAM_BEFORE_ANSWER_VBZ + POS_1_GRAM_BEFORE_ANSWER_WDT + POS_1_GRAM_BEFORE_ANSWER_WP + POS_1_GRAM_BEFORE_ANSWER_WRB + POS_1_GRAM_IN_ANSWER_COUNT_CC + POS_1_GRAM_IN_ANSWER_COUNT_CD + POS_1_GRAM_IN_ANSWER_COUNT_DT + POS_1_GRAM_IN_ANSWER_COUNT_EX + POS_1_GRAM_IN_ANSWER_COUNT_IN + POS_1_GRAM_IN_ANSWER_COUNT_JJ + POS_1_GRAM_IN_ANSWER_COUNT_JJR + POS_1_GRAM_IN_ANSWER_COUNT_JJS + POS_1_GRAM_IN_ANSWER_COUNT_MD + POS_1_GRAM_IN_ANSWER_COUNT_NN + POS_1_GRAM_IN_ANSWER_COUNT_NNP + POS_1_GRAM_IN_ANSWER_COUNT_NNPS + POS_1_GRAM_IN_ANSWER_COUNT_NNS + POS_1_GRAM_IN_ANSWER_COUNT_POS + POS_1_GRAM_IN_ANSWER_COUNT_PRP + POS_1_GRAM_IN_ANSWER_COUNT_RB + POS_1_GRAM_IN_ANSWER_COUNT_RBR + POS_1_GRAM_IN_ANSWER_COUNT_RBS + POS_1_GRAM_IN_ANSWER_COUNT_RP + POS_1_GRAM_IN_ANSWER_COUNT_TO + POS_1_GRAM_IN_ANSWER_COUNT_VB + POS_1_GRAM_IN_ANSWER_COUNT_VBD + POS_1_GRAM_IN_ANSWER_COUNT_VBG + POS_1_GRAM_IN_ANSWER_COUNT_VBN + POS_1_GRAM_IN_ANSWER_COUNT_VBP + POS_1_GRAM_IN_ANSWER_COUNT_VBZ + POS_1_GRAM_IN_ANSWER_COUNT_WDT + POS_1_GRAM_IN_ANSWER_COUNT_WP + POS_1_GRAM_IN_ANSWER_COUNT_WRB + ANSWER_CONTAINS_SRL_A0+ PERCENT_RAW_TOKENS_MATCHING_IN_OUT + PERCENT_TOKENS_IN_ANSWER+ANSWER_ABBREVIATION_WORD_DENSITY + ANSWER_CAPITALIZED_WORD_DENSITY + ANSWER_PRONOMINAL_DENSITY + ANSWER_QUANTIFIER_DENSITY + ANSWER_STOPWORD_DENSITY+ ANSWER_CONTAINS_SRL_A1 + ANSWER_CONTAINS_SRL_A2 + ANSWER_CONTAINS_SRL_A3+ANSWER_CONTAINS_SRL_AM_ADV + ANSWER_CONTAINS_SRL_AM_DIS + ANSWER_CONTAINS_SRL_AM_EXT + ANSWER_CONTAINS_SRL_AM_LOC+ANSWER_CONTAINS_SRL_AM_MNR + ANSWER_CONTAINS_SRL_AM_MOD+ANSWER_CONTAINS_SRL_AM_PNC + ANSWER_CONTAINS_SRL_AM_REC + ANSWER_CONTAINS_SRL_AM_TMP + ANSWER_CONTAINS_SRL_C_A0 + ANSWER_CONTAINS_SRL_C_A1 + ANSWER_CONTAINS_SRL_R_A0+ANSWER_CONTAINS_SRL_R_A1 + ANSWER_CONTAINS_SRL_R_A2 + ANSWER_CONTAINS_SRL_R_AM_LOC + ANSWER_CONTAINS_SRL_R_AM_MNR+ANSWER_CONTAINS_SRL_R_AM_TMP + ANSWER_CONTAINS_SRL_predicate + ANSWER_COVERED_BY_SRL_A0 + ANSWER_COVERED_BY_SRL_A1 + ANSWER_COVERED_BY_SRL_A2+ANSWER_COVERED_BY_SRL_A3 + ANSWER_COVERED_BY_SRL_A4 + ANSWER_COVERED_BY_SRL_AM_ADV + ANSWER_COVERED_BY_SRL_AM_CAU+ANSWER_COVERED_BY_SRL_AM_DIR + ANSWER_COVERED_BY_SRL_AM_DIS + ANSWER_COVERED_BY_SRL_AM_LOC + ANSWER_COVERED_BY_SRL_AM_MNR+ANSWER_COVERED_BY_SRL_AM_PNC + ANSWER_COVERED_BY_SRL_AM_REC + ANSWER_COVERED_BY_SRL_AM_TMP + ANSWER_COVERED_BY_SRL_C_A0+ANSWER_COVERED_BY_SRL_C_A1+ANSWER_COVERED_BY_SRL_predicate' 
    y,X= dmatrices(f,dta,return_type="dataframe")
    y= np.ravel(y)
    model =  LogisticRegression(solver='lbfgs',max_iter=100,penalty='l2')
    model = model.fit(X,y)
    print 
    Qlist=[]
    print '                         ',colored('The Selected Sentences are:','green',attrs=['bold','blink','underline'])
    print
    for sentence,srole in zip(df.Sentence,df.Semantic_Roles):
        sentence=sentence.replace('\'','')
        temp=str(i+1)+' : '+sentence
        print temp
        i=i+1
        qlist=getList(sentence,srole)

        #tokenized representation is used to get ans
        tokens=nltk.word_tokenize(sentence)
        #print 'The candidate gaps are'
	#print

        for (st,end) in qlist:
            ans=' '.join(tokens[st:end+1])
	    ans = re.sub("[!@#$'`]",'', ans)
	    ans=ans.strip()
	    if st<0 or end<0:
	        continue	
	    #print ans
	    anslist.append(ans)
	    semdic=sem_cover_contain(sentence,ans,srole,semdic)
         
	    semdic_list.append(semdic)
	    semdic={}
	    semdic=initialize(semdic) 
            pos=sh.pos1After(sentence,ans)
	    
	    syndic['POS_1_GRAM_AFTER_ANSWER_CC'] = pos == "CC"	#binary	"The first POS tag following the answer span is CC"
            syndic['POS_1_GRAM_AFTER_ANSWER_CD'] = pos == "CD"	#binary	"The first POS tag following the answer span is CD"
            syndic['POS_1_GRAM_AFTER_ANSWER_DT'] = pos == "DT"	#binary	"The first POS tag following the answer span is DT"
            syndic['POS_1_GRAM_AFTER_ANSWER_EX'] = pos == "EX"	#binary	"The first POS tag following the answer span is EX"
            syndic['POS_1_GRAM_AFTER_ANSWER_IN'] = pos == "IN"	#binary	"The first POS tag following the answer span is IN"
            syndic['POS_1_GRAM_AFTER_ANSWER_JJ'] = pos == "JJ"	#binary	"The first POS tag following the answer span is JJ"
            syndic['POS_1_GRAM_AFTER_ANSWER_JJR'] =pos == "JJR"	#binary	"The first POS tag following the answer span is JJR"
            syndic['POS_1_GRAM_AFTER_ANSWER_JJS'] =pos == "JJS"	#binary	"The first POS tag following the answer span is JJS"
            syndic['POS_1_GRAM_AFTER_ANSWER_MD'] = pos == "MD"	#binary	"The first POS tag following the answer span is MD"
            syndic['POS_1_GRAM_AFTER_ANSWER_NN'] = pos == "NN"	#binary	"The first POS tag following the answer span is NN"
            syndic['POS_1_GRAM_AFTER_ANSWER_NNP'] =pos == "NNP"	#binary	"The first POS tag following the answer span is NNP"
            syndic['POS_1_GRAM_AFTER_ANSWER_NNPS']= pos == "NNPS"	#binary	"The first POS tag following the answer span is NNPS"
            syndic['POS_1_GRAM_AFTER_ANSWER_NNS'] =pos == "NNS" 	#binary	"The first POS tag following the answer span is NNS"
            syndic['POS_1_GRAM_AFTER_ANSWER_POS'] =pos == "POS" 	#binary	"The first POS tag following the answer span is POS"
            syndic['POS_1_GRAM_AFTER_ANSWER_PRP'] =pos == "PRP" 	#binary	"The first POS tag following the answer span is PRP"            
            syndic['POS_1_GRAM_AFTER_ANSWER_RB'] = pos == "RB"	#binary	"The first POS tag following the answer span is RB"
            syndic['POS_1_GRAM_AFTER_ANSWER_RBR'] =pos == "RBR" 	#binary	"The first POS tag following the answer span is RBR"
            syndic['POS_1_GRAM_AFTER_ANSWER_RBS'] =pos == "RBS" 	#binary	"The first POS tag following the answer span is RBS"
            syndic['POS_1_GRAM_AFTER_ANSWER_RP'] = pos == "RP"	#binary	"The first POS tag following the answer span is RP"
            syndic['POS_1_GRAM_AFTER_ANSWER_TO'] = pos == "TO"	#binary	"The first POS tag following the answer span is TO"
            syndic['POS_1_GRAM_AFTER_ANSWER_VB'] = pos == "VB"
             	#binary	"The first POS tag following the answer span is VB"
            syndic['POS_1_GRAM_AFTER_ANSWER_VBD'] =pos == "VBD" 	#binary	"The first POS tag following the answer span is VBD"
            syndic['POS_1_GRAM_AFTER_ANSWER_VBG'] =pos == "VBG" 	#binary	"The first POS tag following the answer span is VBG"
            syndic['POS_1_GRAM_AFTER_ANSWER_VBN'] =pos == "VBN" 	#binary	"The first POS tag following the answer span is VBN"
            syndic['POS_1_GRAM_AFTER_ANSWER_VBP'] =pos == "VBP" 	#binary	"The first POS tag following the answer span is VBP"
            syndic['POS_1_GRAM_AFTER_ANSWER_VBZ'] =pos == "VBZ" 	#binary	"The first POS tag following the answer span is VBZ"
            syndic['POS_1_GRAM_AFTER_ANSWER_WDT'] =pos == "WDT" 	#binary	"The first POS tag following the answer span is WDT"
            syndic['POS_1_GRAM_AFTER_ANSWER_WP'] = pos == "WP"	#binary	"The first POS tag following the answer span is WP"
            syndic['POS_1_GRAM_AFTER_ANSWER_WRB'] =pos == "WRB" 	#binary	"The first POS tag following the answer span is WRB"
            pos=sh.pos1Before(sentence,ans)
            syndic['POS_1_GRAM_BEFORE_ANSWER_CC'] =pos == "CC"	#binary	"The first POS tag following the answer span is CC"
            syndic['POS_1_GRAM_BEFORE_ANSWER_CD'] =pos == "CD"	#binary	"The first POS tag following the answer span is CD"
            syndic['POS_1_GRAM_BEFORE_ANSWER_DT'] =pos == "DT"	#binary	"The first POS tag following the answer span is DT"
            syndic['POS_1_GRAM_BEFORE_ANSWER_EX'] =pos == "EX"	#binary	"The first POS tag following the answer span is EX"
            syndic['POS_1_GRAM_BEFORE_ANSWER_IN'] =pos == "IN"	#binary	"The first POS tag following the answer span is IN"
            syndic['POS_1_GRAM_BEFORE_ANSWER_JJ'] =pos == "JJ"	#binary	"The first POS tag following the answer span is JJ"
            syndic['POS_1_GRAM_BEFORE_ANSWER_JJR'] =pos == "JJR"	#binary	"The first POS tag following the answer span is JJR"
            syndic['POS_1_GRAM_BEFORE_ANSWER_JJS'] =pos == "JJS"	#binary	"The first POS tag following the answer span is JJS"
            syndic['POS_1_GRAM_BEFORE_ANSWER_MD'] = pos == "MD"	#binary	"The first POS tag following the answer span is MD"
            syndic['POS_1_GRAM_BEFORE_ANSWER_NN'] = pos == "NN"	#binary	"The first POS tag following the answer span is NN"
            syndic['POS_1_GRAM_BEFORE_ANSWER_NNP'] =pos == "NNP"	#binary	"The first POS tag following the answer span is NNP"
            syndic['POS_1_GRAM_BEFORE_ANSWER_NNPS'] =pos == "NNPS" 	#binary	"The first POS tag following the answer span is NNPS"
            syndic['POS_1_GRAM_BEFORE_ANSWER_NNS'] = pos == "NNS"	#binary	"The first POS tag following the answer span is NNS"
            syndic['POS_1_GRAM_BEFORE_ANSWER_POS'] = pos == "POS"	#binary	"The first POS tag following the answer span is POS"
            syndic['POS_1_GRAM_BEFORE_ANSWER_PRP'] = pos == "PRP"	#binary	"The first POS tag following the answer span is PRP"                 syndic['POS_1_GRAM_AFTER_ANSWER_PRPS'] = sh.pos1After(S,ans,"PRP$")	#binary	"The first POS tag following the answer span is PRP$"
            syndic['POS_1_GRAM_BEFORE_ANSWER_RB'] = pos == "RB"	#binary	"The first POS tag following the answer span is RB"
            syndic['POS_1_GRAM_BEFORE_ANSWER_RBR'] = pos == "RBR"	#binary	"The first POS tag following the answer span is RBR"
            syndic['POS_1_GRAM_BEFORE_ANSWER_RBS'] =pos == "RBS"	#binary	"The first POS tag following the answer span is RBS"
            syndic['POS_1_GRAM_BEFORE_ANSWER_RP'] = pos == "RP"	#binary	"The first POS tag following the answer span is RP"
            syndic['POS_1_GRAM_BEFORE_ANSWER_TO'] = pos == "TO"	#binary	"The first POS tag following the answer span is TO"
            syndic['POS_1_GRAM_BEFORE_ANSWER_VB'] = pos == "VB"	#binary	"The first POS tag following the answer span is VB"
            syndic['POS_1_GRAM_BEFORE_ANSWER_VBD'] =pos == "VBD"	#binary	"The first POS tag following the answer span is VBD"
            syndic['POS_1_GRAM_BEFORE_ANSWER_VBG'] = pos == "VBG"	#binary	"The first POS tag following the answer span is VBG"
            syndic['POS_1_GRAM_BEFORE_ANSWER_VBN'] = pos == "VBN"	#binary	"The first POS tag following the answer span is VBN"
            syndic['POS_1_GRAM_BEFORE_ANSWER_VBP'] = pos == "VBP"	#binary	"The first POS tag following the answer span is VBP"
            syndic['POS_1_GRAM_BEFORE_ANSWER_VBZ'] = pos == "VBZ"	#binary	"The first POS tag following the answer span is VBZ"
            syndic['POS_1_GRAM_BEFORE_ANSWER_WDT'] = pos == "WDT"	#binary	"The first POS tag following the answer span is WDT"
            syndic['POS_1_GRAM_BEFORE_ANSWER_WP'] = pos == "WP"	#binary	"The first POS tag following the answer span is WP"
            syndic['POS_1_GRAM_BEFORE_ANSWER_WRB'] = pos == "WRB"	#binary	"The first POS tag following the answer span is WRB"
            new_dic=sh.pos1In(ans)
            syndic['POS_1_GRAM_IN_ANSWER_COUNT_CC']=new_dic.get('CC',0)#"The number of tokens with POS tag CC in the answer"
      	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_CD']=new_dic.get('CD',0)		#"The number of tokens with POS tag CD in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_DT']=new_dic.get('DT',0)		#"The number of tokens with POS tag DT in the answer"
            syndic['POS_1_GRAM_IN_ANSWER_COUNT_EX']=new_dic.get('EX',0)		#"The number of tokens with POS tag EX in the answer"
            syndic['POS_1_GRAM_IN_ANSWER_COUNT_IN']=new_dic.get('IN',0)		#"The number of tokens with POS tag IN in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_JJ']=new_dic.get('JJ',0)	#"The number of tokens with POS tag JJ in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_JJR']=new_dic.get('JJR',0)	#"The number of tokens with POS tag JJR in the answer"
            syndic['POS_1_GRAM_IN_ANSWER_COUNT_JJS']=new_dic.get('JJS',0)		#"The number of tokens with POS tag JJS in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_MD']=new_dic.get('MD',0)	#"The number of tokens with POS tag MD in the answer"
            syndic['POS_1_GRAM_IN_ANSWER_COUNT_NN']=new_dic.get('NN',0)		#"The number of tokens with POS tag NN in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_NNP']=new_dic.get('NNP',0)	#"The number of tokens with POS tag NNP in the answer"
            syndic['POS_1_GRAM_IN_ANSWER_COUNT_NNPS']=	new_dic.get('NNPS',0)	#"The number of tokens with POS tag NNPS in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_NNS']=new_dic.get('NNS',0)	#"The number of tokens with POS tag NNS in the answer"
            syndic['POS_1_GRAM_IN_ANSWER_COUNT_POS']=new_dic.get('POS',0)	#"The number of tokens with POS tag POS in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_PRP']=new_dic.get('PRP',0)	#"The number of tokens with POS tag PRP in the answer"
	       
            syndic['POS_1_GRAM_IN_ANSWER_COUNT_RB']=new_dic.get('RB',0)		#"The number of tokens with POS tag RB in the answer"
   	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_RBR']=new_dic.get('RBR',0)	#"The number of tokens with POS tag RBR in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_RBS']=new_dic.get('RBS',0)	#"The number of tokens with POS tag RBS in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_RP']=new_dic.get('RP',0)	#"The number of tokens with POS tag RP in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_TO']=new_dic.get('TO',0)	#"The number of tokens with POS tag TO in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_VB']=new_dic.get('VB',0)	#"The number of tokens with POS tag VB in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_VBD']=new_dic.get('VBD',0)	#"The number of tokens with POS tag VBD in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_VBG']=new_dic.get('VBG',0)	#"The number of tokens with POS tag VBG in the answer"
            syndic['POS_1_GRAM_IN_ANSWER_COUNT_VBN']=new_dic.get('VBN',0)		#"The number of tokens with POS tag VBN in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_VBP']=new_dic.get('VBP',0)	#"The number of tokens with POS tag VBP in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_VBZ']=new_dic.get('VBZ',0)	#"The number of tokens with POS tag VBZ in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_WDT']=new_dic.get('WDT',0)	#"The number of tokens with POS tag WDT in the answer"
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_WP']=new_dic.get('WP',0)#"The number of tokens with POS tag WP in the answer"
	       
	    syndic['POS_1_GRAM_IN_ANSWER_COUNT_WRB']=new_dic.get('WRB',0)	#"The number of tokens with POS tag WRB in the answer"
            syntax_list.append(syndic)
	    
	 
	    syndic={}	        
            new_dic={}
	    ns=lh.numberofTokens(sentence)
	    na=lh.numberofTokens(ans)
	    if ns!=0:

	        tokdic['PERCENT_TOKENS_IN_ANSWER']=float(lh.numberofTokens(ans))/lh.numberofTokens(sentence)
	    else:
		tokdic['PERCENT_TOKENS_IN_ANSWER']=0

	    if na!=0:
		
	    	tokdic['PERCENT_RAW_TOKENS_MATCHING_IN_OUT']=float(lh.numberofinout(sentence,ans))/lh.numberofTokens(ans)
	    else:
		tokdic['PERCENT_RAW_TOKENS_MATCHING_IN_OUT']=0
	    token_ct_list.append(tokdic)	
	    tokdic={}
	    anslen=lh.numberofTokens(ans)
	    if(anslen!=0):

                lexdic['ANSWER_CAPITALIZED_WORD_DENSITY']=float(lh.startCapital(ans))/anslen
	        lexdic['ANSWER_ABBREVIATION_WORD_DENSITY']=float(lh.allCapital(ans))/anslen
	        lexdic['ANSWER_PRONOMINAL_DENSITY']=float(lh.pronounCount(ans))/anslen
	        lexdic['ANSWER_STOPWORD_DENSITY']=float(lh.stopCount(ans))/anslen
	        
            else:
		lexdic['ANSWER_CAPITALIZED_WORD_DENSITY']=0
		lexdic['ANSWER_ABBREVIATION_WORD_DENSITY']=0
		lexdic['ANSWER_PRONOMINAL_DENSITY']=0
		lexdic['ANSWER_STOPWORD_DENSITY']=0
	    lexdic['ANSWER_QUANTIFIER_DENSITY']=0
	    lexical_list.append(lexdic)
	    lexdic={}
	finaldic={}
	j=0
	ansdic={}
	for item in semdic_list:
	    finaldic=semdic_list[j].copy()
	    finaldic.update(syntax_list[j])
	    finaldic.update(token_ct_list[j])
	    finaldic.update(lexical_list[j])
	    xstr='POS_1_GRAM_AFTER_ANSWER_CC + POS_1_GRAM_AFTER_ANSWER_CD + POS_1_GRAM_AFTER_ANSWER_DT + POS_1_GRAM_AFTER_ANSWER_EX + POS_1_GRAM_AFTER_ANSWER_IN + POS_1_GRAM_AFTER_ANSWER_JJ + POS_1_GRAM_AFTER_ANSWER_JJR + POS_1_GRAM_AFTER_ANSWER_JJS + POS_1_GRAM_AFTER_ANSWER_MD + POS_1_GRAM_AFTER_ANSWER_NN + POS_1_GRAM_AFTER_ANSWER_NNP + POS_1_GRAM_AFTER_ANSWER_NNPS + POS_1_GRAM_AFTER_ANSWER_NNS + POS_1_GRAM_AFTER_ANSWER_POS + POS_1_GRAM_AFTER_ANSWER_PRP + POS_1_GRAM_AFTER_ANSWER_RB + POS_1_GRAM_AFTER_ANSWER_RBR + POS_1_GRAM_AFTER_ANSWER_RBS + POS_1_GRAM_AFTER_ANSWER_RP + POS_1_GRAM_AFTER_ANSWER_TO + POS_1_GRAM_AFTER_ANSWER_VB + POS_1_GRAM_AFTER_ANSWER_VBD + POS_1_GRAM_AFTER_ANSWER_VBG + POS_1_GRAM_AFTER_ANSWER_VBN + POS_1_GRAM_AFTER_ANSWER_VBP + POS_1_GRAM_AFTER_ANSWER_VBZ + POS_1_GRAM_AFTER_ANSWER_WDT + POS_1_GRAM_AFTER_ANSWER_WP + POS_1_GRAM_AFTER_ANSWER_WRB + POS_1_GRAM_BEFORE_ANSWER_CC + POS_1_GRAM_BEFORE_ANSWER_CD + POS_1_GRAM_BEFORE_ANSWER_DT + POS_1_GRAM_BEFORE_ANSWER_EX + POS_1_GRAM_BEFORE_ANSWER_IN + POS_1_GRAM_BEFORE_ANSWER_JJ + POS_1_GRAM_BEFORE_ANSWER_JJR + POS_1_GRAM_BEFORE_ANSWER_JJS + POS_1_GRAM_BEFORE_ANSWER_MD + POS_1_GRAM_BEFORE_ANSWER_NN + POS_1_GRAM_BEFORE_ANSWER_NNP + POS_1_GRAM_BEFORE_ANSWER_NNPS + POS_1_GRAM_BEFORE_ANSWER_NNS + POS_1_GRAM_BEFORE_ANSWER_POS + POS_1_GRAM_BEFORE_ANSWER_PRP + POS_1_GRAM_BEFORE_ANSWER_RB + POS_1_GRAM_BEFORE_ANSWER_RBR + POS_1_GRAM_BEFORE_ANSWER_RBS + POS_1_GRAM_BEFORE_ANSWER_RP + POS_1_GRAM_BEFORE_ANSWER_TO + POS_1_GRAM_BEFORE_ANSWER_VB + POS_1_GRAM_BEFORE_ANSWER_VBD + POS_1_GRAM_BEFORE_ANSWER_VBG + POS_1_GRAM_BEFORE_ANSWER_VBN + POS_1_GRAM_BEFORE_ANSWER_VBP + POS_1_GRAM_BEFORE_ANSWER_VBZ + POS_1_GRAM_BEFORE_ANSWER_WDT + POS_1_GRAM_BEFORE_ANSWER_WP + POS_1_GRAM_BEFORE_ANSWER_WRB + POS_1_GRAM_IN_ANSWER_COUNT_CC + POS_1_GRAM_IN_ANSWER_COUNT_CD + POS_1_GRAM_IN_ANSWER_COUNT_DT + POS_1_GRAM_IN_ANSWER_COUNT_EX + POS_1_GRAM_IN_ANSWER_COUNT_IN + POS_1_GRAM_IN_ANSWER_COUNT_JJ + POS_1_GRAM_IN_ANSWER_COUNT_JJR + POS_1_GRAM_IN_ANSWER_COUNT_JJS + POS_1_GRAM_IN_ANSWER_COUNT_MD + POS_1_GRAM_IN_ANSWER_COUNT_NN + POS_1_GRAM_IN_ANSWER_COUNT_NNP + POS_1_GRAM_IN_ANSWER_COUNT_NNPS + POS_1_GRAM_IN_ANSWER_COUNT_NNS + POS_1_GRAM_IN_ANSWER_COUNT_POS + POS_1_GRAM_IN_ANSWER_COUNT_PRP + POS_1_GRAM_IN_ANSWER_COUNT_RB + POS_1_GRAM_IN_ANSWER_COUNT_RBR + POS_1_GRAM_IN_ANSWER_COUNT_RBS + POS_1_GRAM_IN_ANSWER_COUNT_RP + POS_1_GRAM_IN_ANSWER_COUNT_TO + POS_1_GRAM_IN_ANSWER_COUNT_VB + POS_1_GRAM_IN_ANSWER_COUNT_VBD + POS_1_GRAM_IN_ANSWER_COUNT_VBG + POS_1_GRAM_IN_ANSWER_COUNT_VBN + POS_1_GRAM_IN_ANSWER_COUNT_VBP + POS_1_GRAM_IN_ANSWER_COUNT_VBZ + POS_1_GRAM_IN_ANSWER_COUNT_WDT + POS_1_GRAM_IN_ANSWER_COUNT_WP + POS_1_GRAM_IN_ANSWER_COUNT_WRB + ANSWER_CONTAINS_SRL_A0+ PERCENT_RAW_TOKENS_MATCHING_IN_OUT + PERCENT_TOKENS_IN_ANSWER+ANSWER_ABBREVIATION_WORD_DENSITY + ANSWER_CAPITALIZED_WORD_DENSITY + ANSWER_PRONOMINAL_DENSITY + ANSWER_QUANTIFIER_DENSITY + ANSWER_STOPWORD_DENSITY+ ANSWER_CONTAINS_SRL_A1 + ANSWER_CONTAINS_SRL_A2 + ANSWER_CONTAINS_SRL_A3+ANSWER_CONTAINS_SRL_AM_ADV + ANSWER_CONTAINS_SRL_AM_DIS + ANSWER_CONTAINS_SRL_AM_EXT + ANSWER_CONTAINS_SRL_AM_LOC+ANSWER_CONTAINS_SRL_AM_MNR + ANSWER_CONTAINS_SRL_AM_MOD+ANSWER_CONTAINS_SRL_AM_PNC + ANSWER_CONTAINS_SRL_AM_REC + ANSWER_CONTAINS_SRL_AM_TMP + ANSWER_CONTAINS_SRL_C_A0 + ANSWER_CONTAINS_SRL_C_A1 + ANSWER_CONTAINS_SRL_R_A0+ANSWER_CONTAINS_SRL_R_A1 + ANSWER_CONTAINS_SRL_R_A2 + ANSWER_CONTAINS_SRL_R_AM_LOC + ANSWER_CONTAINS_SRL_R_AM_MNR+ANSWER_CONTAINS_SRL_R_AM_TMP + ANSWER_CONTAINS_SRL_predicate + ANSWER_COVERED_BY_SRL_A0 + ANSWER_COVERED_BY_SRL_A1 + ANSWER_COVERED_BY_SRL_A2+ANSWER_COVERED_BY_SRL_A3 + ANSWER_COVERED_BY_SRL_A4 + ANSWER_COVERED_BY_SRL_AM_ADV + ANSWER_COVERED_BY_SRL_AM_CAU+ANSWER_COVERED_BY_SRL_AM_DIR + ANSWER_COVERED_BY_SRL_AM_DIS + ANSWER_COVERED_BY_SRL_AM_LOC + ANSWER_COVERED_BY_SRL_AM_MNR+ANSWER_COVERED_BY_SRL_AM_PNC + ANSWER_COVERED_BY_SRL_AM_REC + ANSWER_COVERED_BY_SRL_AM_TMP + ANSWER_COVERED_BY_SRL_C_A0+ANSWER_COVERED_BY_SRL_C_A1+ANSWER_COVERED_BY_SRL_predicate' 
	    X_test=dmatrix(xstr,finaldic,return_type="dataframe")
	    #print X_test.shape
	    #print X_test.columns
	    #print X_test
	    probs = model.predict_proba(X_test)
	    
	    #print probs	
	    
	    finaldic={}
	    ansdic[anslist[j]]=probs[0][1]
	    j=j+1
	try:
		max1=max(ansdic.iteritems(), key=operator.itemgetter(1))[0]
		#print max1
		ansdic.pop(max1, None)
		max2=max(ansdic.iteritems(), key=operator.itemgetter(1))[0]
		q1=sentence.replace(max1,"_________________")
		
	
		Qlist.append(q1)
	except:
		q=''
		
        semdic_list=[]        
	syntax_list=[]
	token_ct_list=[]
	lexical_list=[]
	ansdic={}
	anslist=[]
	print
print '                          ',colored('The Questions Are:','green',attrs=['bold','blink','underline'])
print 
i=0
for q in Qlist:
    i=i+1
    q=q.replace("_________________",colored("_________________",'red'))
    print str(i),':',q
    print 
the_filename=os.path.join(os.path.dirname(__file__),'Questions','Question.txt')   
with open(the_filename, 'wb') as f:
    for s in Qlist:
        f.write(s + '\n')


    
        
	
   



	

