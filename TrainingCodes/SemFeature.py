# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
import ast
import re
import nltk
import nltk
from nltk.tree import *


MAX=2

def getParseDepth(S,ans,tree):

    x=ast.literal_eval(tree)
    x=x[0]
    t=Tree.fromstring(x)


    
    height=t.height()
    ans=' '.join(nltk.word_tokenize(ans))
    for h in range(2,height):
    
        for s in t.subtrees(lambda t: t.height() == h):
            if ' '.join(s.leaves()).find(ans)!=-1:
                return h
                
    
        


def intialize(semdic):
    semdic['ANSWER_PARSE_DEPTH_IN_SRL']=0          #"The constituent parse depth of the answer within its covering SRL"
    semdic['ANSWER_COVERED_BY_SRL_A0']=False        #"Does a span labeled A0 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_A1']=False          #"Does a span labeled A1 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_A2']=False            #"Does a span labeled A2 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_A3']=False        #"Does a span labeled A3 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_A4']=False            #"Does a span labeled A4 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM-ADV']=False        #"Does a span labeled AM-ADV cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM-CAU']=False        #"Does a span labeled AM-CAU cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM-DIR']=False        #"Does a span labeled AM-DIR cover the answer"  
    semdic['ANSWER_COVERED_BY_SRL_AM-DIS']=False        #"Does a span labeled AM-DIS cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM-LOC']=False        #"Does a span labeled AM-LOC cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM-MNR']=False        #"Does a span labeled AM-MNR cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM-PNC']=False        #"Does a span labeled AM-PNC cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM-REC']=False        #"Does a span labeled AM-REC cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_AM-TMP']=False        #"Does a span labeled AM-TMP cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_C-A0']=False          #"Does a span labeled C-A0 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_C-A1']=False          #"Does a span labeled C-A1 cover the answer"
    semdic['ANSWER_COVERED_BY_SRL_predicate']=False     #"Does a span labeled predicate cover the answer"
    semdic['ANSWER_CONTAINS_SRL_A0']=False              #"Is an SRL span labeled A0 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_A1']=False             #"Is an SRL span labeled A1 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_A2']=False              #"Is an SRL span labeled A2 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_A3']=False              #"Is an SRL span labeled A3 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM-ADV']=False          #"Is an SRL span labeled AM-ADV contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM-DIS']=False          #"Is an SRL span labeled AM-DIS contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM-EXT']=False          #"Is an SRL span labeled AM-EXT contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM-LOC']=False          #"Is an SRL span labeled AM-LOC contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM-MNR']=False          #"Is an SRL span labeled AM-MNR contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM-MOD']=False          #"Is an SRL span labeled AM-MOD contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM-PNC']=False          #"Is an SRL span labeled AM-PNC contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM-REC']=False         #"Is an SRL span labeled AM-REC contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_AM-TMP']=False          #"Is an SRL span labeled AM-TMP contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_C-A0']=False            #"Is an SRL span labeled C-A0 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_C-A1']=False           #"Is an SRL span labeled C-A1 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R-A0']=False           #"Is an SRL span labeled R-A0 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R-A1']=False          #"Is an SRL span labeled R-A1 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R-A2']=False             #"Is an SRL span labeled R-A2 contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R-AM-LOC']=False        #"Is an SRL span labeled R-AM-LOC contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R-AM-MNR']=False        #"Is an SRL span labeled R-AM-MNR contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_R-AM-TMP']=False        #"Is an SRL span labeled R-AM-TMP contained within the span of the answer"
    semdic['ANSWER_CONTAINS_SRL_predicate']=False       #"Is an SRL span labeled predicate contained within the span of the answer"
    '''
    semdic['ANSWER_NAMED_ENTITY_DENSITY']=0         #"Percentage of tokens in the answer that have a named entity label"
    semdic['SENTENCE_NAMED_ENTITY_DENSITY']=0       #"Percentage of tokens in the sentence that have a named entity label"
    semdic['NUM_NAMED_ENTITIES_IN_ANSWER']=0        #"Number of tokens in the answer that have a named entity label"
    semdic['NUM_NAMED_ENTITIES_OUT_ANSWER']=0       #"Number of tokens outside of the answer that have a named entity label"
    semdic['PERCENT_OF_NAMED_ENTITIES_IN_ANSWER']=0.0 #"Percentage of the sentence named entity tokens found in the answer"
    semdic['NAMED_ENTITY_IN_ANSWER_COUNT_PERS']=0   #"Does the answer contain a PERS named entity?"
    semdic['NAMED_ENTITY_IN_ANSWER_COUNT_ORG']=0    #"Does the answer contain a ORG named entity?"
    semdic['NAMED_ENTITY_IN_ANSWER_COUNT_LOC']=0    #"Does the answer contain a LOC named entity?"
    semdic['NAMED_ENTITY_OUT_ANSWER_COUNT_ORG']=0   #"Does the text outside of the answer contain a PERS named entity?"
    semdic['NAMED_ENTITY_OUT_ANSWER_COUNT_PERS']=0  #"Does the text outside of the answer contain a ORG named entity?"
    semdic['NAMED_ENTITY_OUT_ANSWER_COUNT_LOC']=0   #"Does the text outside of the answer contain a LOC named entity?"
    '''
    return semdic


def splitkeepsep(s, sep):
    return reduce(lambda acc, elem: acc[:-1] + [acc[-1] + elem] if elem == sep else acc + [elem], re.split("(%s)" % re.escape(sep), s), [])

        
def sem_cover_contain(S,ans,dic_tags,semdic):

    x=dic_tags['Semantic_Roles']
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



    
if __name__ == '__main__':
    
    
    dtf = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'Train.csv'))
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'MindTheGap-1.0-Data.tsv'), header=0, \
                    delimiter="\t", quoting=3)
    







    print "Data available for a sentence"
    print dtf.dtypes
    print

    dic_dic={}
    for Sid,sr,Ct in zip(dtf.SentenceID,dtf.Semantic_Roles,dtf.Constituency_Tree):
            dic={}
            dic['Semantic_Roles']=sr
            dic['Constituency_Tree']=Ct
            dic_dic[Sid]=dic
    print len(dic_dic)

    
    

        
    
    i=0
    semdic_list=[]
    semdic={}
    semdic=intialize(semdic)
    ct=0
    
    for Sid,S,ans,ques,Jdg,Jid,Qid in zip(data.SentenceID,data.Sentence,data.Answer,data.Question,data.Judgment,data.JudgeId,data.QuestionID):
        
        if Qid=='None':
            continue

        if i==0:            
            semdic=sem_cover_contain(S,ans,dic_dic[Sid],semdic) 
            semdic['QuestionID']=Qid
            semdic['ANSWER_PARSE_DEPTH_IN_SRL']=getParseDepth(S,ans,dic_dic[Sid]['Constituency_Tree'])
            ct = ct + 1
        if i==3:
             semdic_list.append(semdic)
             semdic={}
             semdic=intialize(semdic)
             
        i=(i+1)%4
         
    

    dta=pd.DataFrame.from_dict(semdic_list, orient='columns')
    dta.to_csv(os.path.join(os.path.dirname(__file__), 'data','semantic_feature.csv'), index=False)

     
          