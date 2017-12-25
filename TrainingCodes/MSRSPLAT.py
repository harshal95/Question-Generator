import requests

language = "en"
analyzers = "Constituency_Tree,Labeled_Dependency_Tree,Semantic_Roles"
sentences=["He is a good boy","American football  is a sport played by two teams of eleven players on a rectangular field with goalposts at each end."]

request = "http://msrsplat.cloudapp.net/SplatServiceJson.svc/Analyze?language="+language+"&analyzers="+analyzers\
+"&appId=358378CB-9C65-43AD-A365-FF7D6AD85620&json=x&input="

for sentence in sentences: 
	print "Source_Sentence:",sentence
	sentence_request=request+sentence

	r=requests.get(sentence_request);
	results=r.json()
	for item in results:
		print item['Key']
		print item['Value']
		print

	print 
