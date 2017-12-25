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



def splitkeepsep(s, sep):
    return reduce(lambda acc, elem: acc[:-1] + [acc[-1] + elem] if elem == sep else acc + [elem], re.split("(%s)" % re.escape(sep), s), [])

LANGUAGE =  "english"
SENTENCES_COUNT = 10


if __name__ == '__main__':



	
	
	parser = PlaintextParser.from_file("passage.txt", Tokenizer(LANGUAGE))
	stemmer = Stemmer(LANGUAGE)

	summarizer = Summarizer(stemmer)
	summarizer.stop_words = get_stop_words(LANGUAGE)
	i=0
	for sentence in summarizer(parser.document, SENTENCES_COUNT):
		i=i+1
		print str(i),':',(sentence)
		print 

	'''
	f = open('passage.txt','r')

	text=f.read()
	print text

	SS=splitkeepsep(text,'. ')
	
	for s in SS:
		print s
		print

	height=t.height()
	for h in range(2,height):
	print 'level:',str(h)
	for s in t.subtrees(lambda t: t.height() == h):
	print ' '.join(s.leaves())
	print 
	print
	'''
