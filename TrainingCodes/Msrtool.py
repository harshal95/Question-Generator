import requests
def getList(analyzers,sentence,Sid):

	language = "en"
	request = "http://msrsplat.cloudapp.net/SplatServiceJson.svc/Analyze?language="+language+"&analyzers="+analyzers\
	+"&appId=358378CB-9C65-43AD-A365-FF7D6AD85620&json=x&input="

	sentence_request=request+sentence

	r=requests.get(sentence_request);
	results=r.json()
	dic={}
	dic['SentenceID']=Sid
	dic['Sentence']=sentence
	for item in results:
		 dic[item['Key']]=item['Value']

	return dic
		

	
	
