import csv
import os
import math
from util import helper, textprocessing
import sys
from pathlib import Path


def function():

	# reading query files
	#tsv_file = open("/home/cse/phd/csz208507/scratch/MSMARCO-DocRanking/msmarco-docdev-queries.tsv")

	if (len(sys.argv)!=5):
		print("wrong command")
		print("command : prob_rerank [query-file] [top-100-file] [collection-file] [expansion-limit]")
		return 0
	
	if(Path(sys.argv[1]).exists()==False):
		print("query-file does not exist")
		return 0

	tsv_file = open(sys.argv[1])
	read_tsv = csv.reader(tsv_file,delimiter="\t")

	query_id = []
	query_str = []
	top100_doc = {}
	# Iterating over queries
	for read in read_tsv:
		query_id.append( read[0])
		query_str.append(read[1])
		top100_doc[read[0]]=[]
		#break
	tsv_file.close()

	# preprocessing the top100 docs
	if(Path(sys.argv[2]).exists()==False):
		print("top-100-files does not exist")
		return 0
	
	top100_file = sys.argv[2]
	top100_doc = helper.get_top100(top100_file,top100_doc)

	#print(top100_doc[query_id[0]])
	#print(top100_doc[query_id[1]])

	# reading the doc file and preprocessing it
	if(Path(sys.argv[3]).exists()==False):
		print("collection-file does not exist")
		return 0
	doc_path = sys.argv[3]
	#print(helper.get_doc(doc_path))
	#print(helper.get_doc_offset(doc_path))
	doc_offset = helper.get_doc_offset(doc_path)


	# load the stopword file
	stopwords = textprocessing.read_stopwords("resources/stopwords_en.txt")
	#print(stopwords)

	if os.path.isdir(os.path.join(os.getcwd(),"output")) == False:
		os.mkdir(os.path.join(os.getcwd(),"output"))
	else:
		folder = os.path.join(os.getcwd(),"output")
		for filename in os.listdir(folder):
			file_path = os.path.join(folder, filename)
			try:
				if os.path.isfile(file_path) or os.path.islink(file_path):
					os.unlink(file_path)
			except Exception as e:
				print('Failed to delete %s. Reason: %s' % (file_path, e))

	for i in range(len(query_id)):
		q_id = query_id[i]
		q_str = textprocessing.preprocess_text( query_str[i],stopwords)
		q_top100_doc = top100_doc[q_id]
		#print(q_top100_doc)
	
		for ii in range(len(q_top100_doc)):
			#print(q_top100_doc[ii])
			q_top100_doc[ii][1] = float(q_top100_doc[ii][1])
		
	
		# get the relevent documents from the corpus
		subset_doc = helper.get_100doc(q_top100_doc, doc_offset,doc_path)
		#print(subset_doc)
	
		doc_dict = {}
		doc_dict1 = {}
		dlav = 0
		#clean the title and the body of the documents retrieved
		for doc in subset_doc:
			doc_id = doc[0]
			#word_list = textprocessing.preprocess_text(doc[1], stopwords)+textprocessing.preprocess_text(doc[2], stopwords)
			word_list = textprocessing.preprocess_text(doc[1],stopwords)
			#print(doc_id)
			doc_dict1[doc_id] = word_list
			word_set = set(word_list)
			#print(word_set)
			doc_dict[doc_id]=list(word_set)
			#print(len(word_set))
			dlav = dlav + len(word_set)
			#print(dlav)

		dlav = dlav / len(q_top100_doc)
	
		inv_index = helper.create_inverted_index(doc_dict)
		#print(" ")
		#print(inv_index)
	
		folder = os.path.join(os.getcwd(),"output")

		
		# sort the top documents
		q_top100_doc = sorted(q_top100_doc,key=lambda x:int(x[1]),reverse=True)
		#print(" ")
		#print(q_top100_doc)
		
		
		
		for tx in range(int(sys.argv[4])):
			relev_doc = 10
	
			# we are assuming that the top 10 docs are relevent so the terms in these docs will be
			list_top_word = []
			for ii in range(relev_doc):
				list_top_word = list_top_word + doc_dict[q_top100_doc[ii][0]]
			list_top_word = list(set(list_top_word))
			#print(" ")
			#print(list_top_word)
	

			# creating the inverted index of the top documents
			min_inv_index = helper.create_min_inverted_index(list_top_word, doc_dict,relev_doc,q_top100_doc)
			#print(min_inv_index)
	
			dic_top_words = []
			for wd in list_top_word:
				vri = len(min_inv_index[wd])
				#print(inv_index[wd]["pi"])
				#print(inv_index[wd]["ui"])
				wi = math.log(inv_index[wd]["pi"]/(1-inv_index[wd]["pi"]))+math.log((1-inv_index[wd]["ui"])/(inv_index[wd]["ui"])) 
				dic_top_words.append([wd,vri*wi])
	
			dic_top_words = sorted( dic_top_words,key=lambda x:x[1],reverse=True)
			#print( dic_top_words)
	
			new_word=""
			for word,weight in dic_top_words:
				if word in q_str:
					continue
				else:
					new_word = word
					#print(q_str)
					q_str.append(word)
					#print (q_str)
					break
	
			#### code to rerank the documents using BM25
			#print(q_top100_doc)
			q_top100_doc = helper.calculate_top100_bm25(q_top100_doc,new_word,inv_index,dlav,doc_dict1)	
			#print(q_top100_doc)
		
			q_top100_doc = sorted(q_top100_doc,key=lambda x:int(x[1]),reverse=True)
		
			### writing first output to the file
			f=open(os.path.join(folder,"score_"+str(tx+1)), "a")
		
			for j in range(len(q_top100_doc)):
				f.write(q_id+" Q0 "+str(q_top100_doc[j][0])+" "+ str(j+1)+" "+str(q_top100_doc[j][1])+" IndriQueryLikelihood\n")
		
			f.close()	

			### code to recalculate pi and ui for each term
			inv_index = helper.calculate_pi_ui(min_inv_index,relev_doc,inv_index,len(q_top100_doc))	

			### loop ends

if __name__ == "__main__":
	function()
