import pandas as pd
import matplotlib.pylab as pl
import numpy as np
import os
import ast
import nltk
import sys
from nltk.corpus import stopwords 
'''
def pos1After2(S,ans,str_check):
    sent_tok = nltk.pos_tag(nltk.word_tokenize(S))
    ans_tok  = nltk.pos_tag(nltk.word_tokenize(ans))
    index = sent_tok.index(ans_tok[-1])
    str_ans = sent_tok[index + 1][1]
    return str_ans == str_check
'''
    
def pos1After(S,ans):
    try:
        index = S.find(ans)
        index = index + len(ans)
        if index >= len(S):
            return "false"
        while S[index]==',' or S[index] == ';' or S[index] ==':' or S[index] =='!' or S[index] == '?' or S[index]==" " or S[index] =="." :
            index = index + 1
            if index >= len(S):
                return "false"
        
        
        new_ans = S[index:]
        new_ans = new_ans.split(" ")
        str_ans = nltk.pos_tag(new_ans[0])
        return str_ans[0][1] 
    except:
	return "false"
        
    
def pos1Before(S,ans):
    try:
        index = S.find(ans)
        if index == 0:
            return "false"
        index =index - 1
        while S[index]==',' or S[index] == ';' or S[index] ==':' or S[index] =='!' or S[index] == '?' or S[index]==" " or S[index] =="." :
            index = index - 1
            if index ==0:
                return "false"
        #print index
        new_ans = S[0:index]
        new_ans = new_ans.split(" ")
        str_ans = nltk.pos_tag(new_ans[-1])
        return str_ans[0][1]

    except:
        return "false"

        

def pos1In(ans):
    try:
        ans_tok = nltk.pos_tag(nltk.word_tokenize(ans))
        mydic={}
        for(x,y) in ans_tok:
            if mydic.has_key(y) == False:
                mydic[y]=1
            else:
                mydic[y]=mydic[y]+1
        return mydic
    except:
	print "Unexpected error:", sys.exc_info()
	mydic={}	
	return mydic
        
        
        
        
    
    
    
    


S = "A few fudai daimyo, such as the Ii of Hikone , held large han, but many were small."
ans = "large han"
str_check = "CC"
#print pos1Before(S,ans,str_check)

