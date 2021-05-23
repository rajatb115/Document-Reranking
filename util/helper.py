import csv
import os
import math
import pandas as pd
from datetime import datetime

# Function to compute current time
def get_current_time():
	now = datetime.now()
	return (now.strftime("%H:%M:%S"))

# retrive the relevent document regarding the query
def get_top100(filepath,dic):
	current_time = get_current_time()
	print("get top 100 doc list start: "+str(current_time))
	file_text = open(filepath)
	read_text = file_text.readline()
	while(read_text):
		read_text = read_text.strip()
		#print(read_text)
		read = read_text.split(" ")
		dic[read[0]].append([read[2],read[4]])
		#print(dic[read[0]])
		read_text = file_text.readline()
		#break
	current_time = get_current_time()
	print("get top 100 doc list end: "+str(current_time))
	file_text.close()
	return dic

# using pandas its taking 14 mins to read and generate index for whole document	
def get_doc(filepath):
	j=0
	dict_docid={}
	
	#current_time = get_current_time()
	#print("starting time of retrival: "+str(current_time))
	for chunk in pd.read_csv(filepath,sep="\t",chunksize=10):
		for i in range(len(chunk)):
			dict_docid[chunk.iloc[i,0]]=j
			j+=1
			#print(chunk.iloc[i,0]+" "+str(dict_docid[chunk.iloc[i,0]]))
			#print(chunk.iloc[i, 0], chunk.iloc[i, 1])
	#current_time = get_current_time()
	#print("Ending time of retrival: "+str(current_time))
	return dict_docid

# using line by line retrival of file its taking 4 mins total
def get_doc_offset(filepath):
	dict_docid={}
	#current_time = get_current_time()
	#print("starting time of retrival: "+str(current_time))
	if (os.path.exists(filepath)):
		file1 = open(filepath,"r")
		pos = file1.tell()
		line = file1.readline()
		i =0
	
		while(line):
			#print (line.split("\t"))
			line = line.strip()
			#print(line)
			dict_docid[line.split("\t")[0]]=pos
			pos = file1.tell()
			line = file1.readline()
			#pos = file1.tell()
			#break
		file1.close()
	#current_time = get_current_time()
	#print("Ending time of retrival: "+str(current_time))
	return dict_docid
	
def get_100doc(q_top100_doc,doc_offset,filepath):
	#current_time = get_current_time()
	#print("starting time of doc retrival: "+str(current_time))
	
	file1 = open(filepath,"r")
	
	doc_list = []	

	for doc_id,doc_weight in q_top100_doc:
		offset = doc_offset[doc_id]
		file1.seek(offset)
		data_doc = file1.readline().strip().split("\t")
		lis = []
		l = len(data_doc)
		lis.append(data_doc[0])
		lis.append(data_doc[l-1])
		doc_list.append(lis)
	file1.close()
	#current_time = get_current_time()
	#print("ending time of doc retrival: "+str(current_time))
	return doc_list

# function to create inverted index
def create_inverted_index(documents):
	lis = []
	for i,l in documents.items():
		lis = lis + l
	lis = list(set(lis))
	
	dic = {}
	for wd in lis:
		dic[wd]={}
		dic[wd]["posting_list"]=[]
		dic[wd]["pi"]=0.5
		dic[wd]["ui"]=0.0
		
	
	for i,l in documents.items():
		for wd in l:
			dic[wd]["posting_list"].append(i)
	
	for word in dic:
		#print(word)
		dic[word]["ui"] = len(dic[word]["posting_list"])/(len(documents)+1)
		
	return dic


def create_min_inverted_index(words,doc_dic,relev_doc,q_top100_doc):
	inv_ind={}
	for w in words:
		inv_ind[w]=[]
	#for doc_id, doc_score in q_top100_doc:
		#print (str(doc_id)+" " +str(doc_score))
	for i in range(relev_doc):
		doc_id = q_top100_doc[i][0]
		for word in doc_dic[doc_id]:
			inv_ind[word].append(doc_id)
	return inv_ind



def calculate_pi_ui(min_inv_index,vr,inv_index,N):
	# Let k =5
	k=5
	for word,value in min_inv_index.items():
		vri = len(value)
		dfi = len(inv_index[word]["posting_list"])
		inv_index[word]["ui"] = (abs(dfi-vri)+0.5)/(abs(N-vr)+1)
		inv_index[word]["pi"] = (abs(vri)+k*inv_index[word]["pi"])/((abs(vr))+k)
	return inv_index


def calculate_top100_bm25(q_top100_doc,new_word,inv_index,dlav,doc_dict):
	k1 =  1.5
	b = 0.75
	wt_new_word =  math.log(inv_index[new_word]["pi"]/(1-inv_index[new_word]["pi"]))+math.log((1-inv_index[new_word]["ui"])/(inv_index[new_word]["ui"]))
	for doc in inv_index[new_word]["posting_list"]:
		tfi = doc_dict[doc].count(new_word)
		#print(tfi)
		dl = len(doc_dict[doc])
		for i in range(len(q_top100_doc)):
			if q_top100_doc[i][0] == doc:
				q_top100_doc[i][1] =q_top100_doc[i][1] +  wt_new_word *((tfi*(1+k1))/(k1*(1-b)+b*(dl/dlav)+tfi))
				break
	return q_top100_doc

 
