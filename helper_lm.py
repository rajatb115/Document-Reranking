

def find_ftic(doc_dict1,word):
	cnt =0
	for doc,lis in doc_dict1.items():
		#print(doc)
		#print(lis)
		cnt+=lis.count(word)
	return cnt

		
