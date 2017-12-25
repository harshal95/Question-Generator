import nltk
from nltk.corpus import stopwords 
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
def getRating(Jdg):
	Jdg=str(Jdg)
	if Jdg=="Good":
		return 1.0
	elif Jdg=="Bad":
		return -1.0
	else:
		return 0.5
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
	
def getProperString(S):
	token=nltk.word_tokenize(S)
	S=' '.join(token)
	return S

def getProperList(S):
	token=nltk.word_tokenize(S)
	return token

def pronounCount(S):
	
	try:
		res=nltk.pos_tag(nltk.word_tokenize(S))
		ct=0
		for (x,y) in res :
			if y=='PRP' or y=='PRP$':
				 ct=ct+1

		return ct
	except:
		print 'err occured'
def stopCount(S):

	stop=stopwords.words('english')
	ct=0

	for word in S:
		if word.lower() in stop:
			ct+=1
	return ct



