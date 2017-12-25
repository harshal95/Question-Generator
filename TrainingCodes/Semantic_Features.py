import pandas as pd
import os
import ast
import nltk
from nltk.tree import Tree
from bllipparser import RerankingParser
from practnlptools.tools import Annotator
from nltk.tree import ParentedTree
import unicodedata

def isCovered(S,ans,annotator):
	
	token=nltk.word_tokenize(S)
	S=' '.join(token)
	3S=unicodedata.normalize('NFKD',S).encode('ascii','ignore')
	#rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
	srl=annotator.getAnnotations(S)['srl']
	print srl



if __name__ == '__main__':

	dtf = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'Train.csv'))
	data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'MindTheGap-1.0-Data.tsv'), header=0, \
	delimiter="\t", quoting=3)



	annotator=Annotator()
	print dtf.dtypes

	#POS TAGS List
	smldic={}
	for Sid,sem in zip(dtf.SentenceID,dtf.Semantic_Roles):

		x=ast.literal_eval(sem)
		smldic[Sid]=x[0] 

	sem_dic={}
	sem_list=[]
	i=0
	ct=0


	for Sid,S,ans,ques,Jdg,Jid in zip(data.SentenceID,data.Sentence,data.Answer,data.Question,data.Judgment,data.JudgeId):

		
		if i==0:
			print 'Line:',str(ct)
			isCovered(S,ans,annotator)
			print smldic[Sid]
			sem_dic['ANSWER_PARSE_DEPTH_IN_SRL'] = 0	#int	 "The constituent parse depth of the answer within its covering SRL"
			sem_dic['ANSWER_COVERED_BY_SRL_A0'] = 0		#binary	"Does a span labeled A0 cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_A1'] = 0		#binary	"Does a span labeled A1 cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_A2'] = 0		#binary	"Does a span labeled A2 cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_A3'] = 0		#binary	"Does a span labeled A3 cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_A4'] = 0		#binary	"Does a span labeled A4 cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_AM-ADV'] = 0		#binary	"Does a span labeled AM-ADV cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_AM-CAU'] = 0		#binary	"Does a span labeled AM-CAU cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_AM-DIR'] = 0		#binary	"Does a span labeled AM-DIR cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_AM-DIS'] = 0		#binary	"Does a span labeled AM-DIS cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_AM-LOC'] = 0		#binary	"Does a span labeled AM-LOC cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_AM-MNR'] = 0		#binary	"Does a span labeled AM-MNR cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_AM-PNC'] = 0		#binary	"Does a span labeled AM-PNC cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_AM-REC'] = 0		#binary	"Does a span labeled AM-REC cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_AM-TMP'] = 0		#binary	"Does a span labeled AM-TMP cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_C-A0'] = 0		#binary	"Does a span labeled C-A0 cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_C-A1'] = 0		#binary	"Does a span labeled C-A1 cover the answer"
			sem_dic['ANSWER_COVERED_BY_SRL_predicate'] = 0		#binary	"Does a span labeled predicate cover the answer"
			sem_dic['ANSWER_CONTAINS_SRL_A0'] = 0		#binary	"Is an SRL span labeled A0 contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_A1'] = 0		#binary	"Is an SRL span labeled A1 contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_A2'] = 0		#binary	"Is an SRL span labeled A2 contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_A3'] = 0		#binary	"Is an SRL span labeled A3 contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_AM-ADV'] = 0		#binary	"Is an SRL span labeled AM-ADV contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_AM-DIS'] = 0		#binary	"Is an SRL span labeled AM-DIS contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_AM-EXT'] = 0		#binary	"Is an SRL span labeled AM-EXT contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_AM-LOC'] = 0		#binary	"Is an SRL span labeled AM-LOC contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_AM-MNR'] = 0		#binary	"Is an SRL span labeled AM-MNR contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_AM-MOD'] = 0	#binary	"Is an SRL span labeled AM-MOD contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_AM-PNC'] = 0	#binary	"Is an SRL span labeled AM-PNC contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_AM-REC'] = 0	#binary	"Is an SRL span labeled AM-REC contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_AM-TMP'] = 0	#binary	"Is an SRL span labeled AM-TMP contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_C-A0'] = 0	#binary	"Is an SRL span labeled C-A0 contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_C-A1'] = 0	#binary	"Is an SRL span labeled C-A1 contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_R-A0'] = 0	#binary	"Is an SRL span labeled R-A0 contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_R-A1'] = 0	#binary	"Is an SRL span labeled R-A1 contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_R-A2'] = 0	#binary	"Is an SRL span labeled R-A2 contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_R-AM-LOC'] = 0	#binary	"Is an SRL span labeled R-AM-LOC contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_R-AM-MNR'] = 0	#binary	"Is an SRL span labeled R-AM-MNR contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_R-AM-TMP'] = 0	#binary	"Is an SRL span labeled R-AM-TMP contained within the span of the answer"
			sem_dic['ANSWER_CONTAINS_SRL_predicate'] = 0	#binary	"Is an SRL span labeled predicate contained within the span of the answer"
			sem_dic['ANSWER_NAMED_ENTITY_DENSITY'] = 0	#float	"Percentage of tokens in the answer that have a named entity label"
			sem_dic['SENTENCE_NAMED_ENTITY_DENSITY'] = 0	#float	"Percentage of tokens in the sentence that have a named entity label"
			sem_dic['NUM_NAMED_ENTITIES_IN_ANSWER'] = 0	#int	"Number of tokens in the answer that have a named entity label"
			sem_dic['NUM_NAMED_ENTITIES_OUT_ANSWER'] = 0	#int	"Number of tokens outside of the answer that have a named entity label"
			sem_dic['PERCENT_OF_NAMED_ENTITIES_IN_ANSWER'] = 0	#float	"Percentage of the sentence named entity tokens found in the answer"
			sem_dic['NAMED_ENTITY_IN_ANSWER_COUNT_PERS'] = 0	#binary	"Does the answer contain a PERS named entity?"
			sem_dic['NAMED_ENTITY_IN_ANSWER_COUNT_ORG'] = 0	#binary	"Does the answer contain a ORG named entity?"
			sem_dic['NAMED_ENTITY_IN_ANSWER_COUNT_LOC'] = 0	#binary	"Does the answer contain a LOC named entity?"
			sem_dic['NAMED_ENTITY_OUT_ANSWER_COUNT_ORG'] = 0	#binary	"Does the text outside of the answer contain a PERS named entity?"
			sem_dic['NAMED_ENTITY_OUT_ANSWER_COUNT_PERS'] = 0	#binary	"Does the text outside of the answer contain a ORG named entity?"
			sem_dic['NAMED_ENTITY_OUT_ANSWER_COUNT_LOC'] = 0	#binary	"Does the text outside of the answer contain a LOC named entity?"
			ct = ct + 1

		if i==3:
			sem_list.append(sem_dic)
			sem_dic={}

		i=(i+1)%4

		if ct >0:
			break
