import csv
import os
import math
from util import helper_lm, helper,textprocessing
import sys
from pathlib import Path

def function_unigram():
	
	# reading query files
	if(Path(sys.argv[1]).exists()==False):
		print("query-file does not exist")
		return 0
	
	#tsv_file = open("/home/cse/phd/csz208507/scratch/MSMARCO-DocRanking/msmarco-docdev-queries.tsv")
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
	
	#top100_file = "/home/cse/phd/csz208507/scratch/MSMARCO-DocRanking/msmarco-docdev-top100"
	top100_file = sys.argv[2]
	top100_doc = helper.get_top100(top100_file,top100_doc)

	# reading the doc file and preprocessing it
	if(Path(sys.argv[3]).exists()==False):
		print("collection-file does not exist")
		return 0
	
	#doc_path = "/home/cse/phd/csz208507/scratch/MSMARCO-DocRanking/msmarco-docs.tsv"
	doc_path = sys.argv[3]
	doc_offset = helper.get_doc_offset(doc_path)


	# load the stopword file
	stopwords = textprocessing.read_stopwords("/home/cse/phd/csz208507/IR2/resources/stopwords_en.txt")
	#print(stopwords)

	if os.path.isdir(os.path.join(os.getcwd(),"output_lm_uni")) == False:
		os.mkdir(os.path.join(os.getcwd(),"output_lm_uni"))
	else:
		folder = os.path.join(os.getcwd(),"output_lm_uni")
		for filename in os.listdir(folder):
			file_path = os.path.join(folder, filename)
			try:
				if os.path.isfile(file_path) or os.path.islink(file_path):
					os.unlink(file_path)
			except Exception as e:
				print('Failed to delete %s. Reason: %s' % (file_path, e))

	for i in range(len(query_id)):
		q_id = query_id[i]
		q_str = textprocessing.preprocess_text(query_str[i],stopwords)
		q_top100_doc = top100_doc[q_id]
		
		q_final_weight = []
		for ii in range(len(q_top100_doc)):
			lt=[]
			lt.append(q_top100_doc[ii][0])
			lt.append(0)
			q_final_weight.append(lt)
	
	
		subset_doc = helper.get_100doc(q_top100_doc, doc_offset,doc_path)
	
		doc_dict1 = {}
		dc = 0
		# clean the body of the documents retrieved
		for doc in subset_doc:
			doc_id = doc[0]
			word_list = textprocessing.preprocess_text(doc[1],stopwords)
			doc_dict1[doc_id] = word_list
			dc=dc+len(word_list)
		
		folder = os.path.join(os.getcwd(),"output_lm_uni")
		mew = 1.5
		for word in q_str:
			for jj in range(len(q_final_weight)):
				ftid = doc_dict1[q_final_weight[jj][0]].count(word)
				#print(ftid)
				pcti = helper_lm.find_ftic(doc_dict1,word)/dc
				if ftid == 0 and pcti == 0:
					continue
				q_final_weight[jj][1] = q_final_weight[jj][1] + math.log((ftid+mew*pcti)/(len(doc_dict1[q_final_weight[jj][0]])+mew))
	
		#for word,weight in q_final_weight:
			#print(word," ",weight)
	
		q_final_weight = sorted(q_final_weight,key=lambda x:int(x[1]),reverse=True)
		
		### writing first output to the file
		f=open(os.path.join(folder,"score_lm_unigram"), "a")
		
		for j in range (len(q_final_weight)):
			f.write(q_id+" Q0 "+str(q_final_weight[j][0])+" "+ str(j+1)+" "+str(q_final_weight[j][1])+" IndriQueryLikelihood\n")
		f.close()
	
def function_bigram():
	#print("bigram")
	# reading query files
	if(Path(sys.argv[1]).exists()==False):
		print("query-file does not exist")
		return 0
	
	#tsv_file = open("/home/cse/phd/csz208507/scratch/MSMARCO-DocRanking/msmarco-docdev-queries.tsv")
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

	#top100_file = "/home/cse/phd/csz208507/scratch/MSMARCO-DocRanking/msmarco-docdev-top100"
	top100_file = sys.argv[2]
	top100_doc = helper.get_top100(top100_file,top100_doc)

	# reading the doc file and preprocessing it
	if(Path(sys.argv[3]).exists()==False):
		print("collection-file does not exist")
		return 0
	
	#doc_path = "/home/cse/phd/csz208507/scratch/MSMARCO-DocRanking/msmarco-docs.tsv"
	doc_path = sys.argv[3]
	doc_offset = helper.get_doc_offset(doc_path)

	# load the stopword file
	stopwords = textprocessing.read_stopwords("/home/cse/phd/csz208507/IR2/resources/stopwords_en.txt")
	#print(stopwords)

	if os.path.isdir(os.path.join(os.getcwd(),"output_lm_bi")) == False:
		os.mkdir(os.path.join(os.getcwd(),"output_lm_bi"))
	else:
		folder = os.path.join(os.getcwd(),"output_lm_bi")
		for filename in os.listdir(folder):
			file_path = os.path.join(folder, filename)
			try:
				if os.path.isfile(file_path) or os.path.islink(file_path):
					os.unlink(file_path)
			except Exception as e:
				print('Failed to delete %s. Reason: %s' % (file_path, e))


	for i in range(len(query_id)):
		q_id = query_id[i]
		q_str = textprocessing.preprocess_text(query_str[i],stopwords)
		q_top100_doc = top100_doc[q_id]
		
		
		q_final_weight = []
		for ii in range(len(q_top100_doc)):
			lt=[]
			lt.append(q_top100_doc[ii][0])
			lt.append(0)
			q_final_weight.append(lt)

		subset_doc = helper.get_100doc(q_top100_doc, doc_offset,doc_path)
		
		doc_dict1 = {}
		doc_dict2 = {}
		dc = 0
		# clean the body of the documents retrieved
		for doc in subset_doc:
			doc_id = doc[0]
			word_list = textprocessing.preprocess_text(doc[1],stopwords)
			
			bigm = []
			for ii in range(1,len(word_list),1):
				bigm.append(str(word_list[ii-1])+" "+str(word_list[ii]))
			
			doc_dict1[doc_id] = word_list
			dc=dc+len(word_list)
			doc_dict2[doc_id] = bigm
		
		folder = os.path.join(os.getcwd(),"output_lm_bi")	
		
		mew1 = 1.5
		mew2 = 5
		for ii in range(len(q_str)):
			q1 = q_str[ii]
			if ii != 0 :
				q1 = q_str[ii-1]+" "+q1
			for j in range(len(q_final_weight)):
				if(len(doc_dict1[q_final_weight[j][0]])>0):
					lam1 = mew1/(mew1+len(doc_dict1[q_final_weight[j][0]]))
					unigm = lam1*(doc_dict1[q_final_weight[j][0]].count(q_str[ii])/(len(doc_dict1[q_final_weight[j][0]])))+ (1-lam1)/3213835
				else:
					unigm = 1/3213835
				#lam2 = mew2/(mew2+len(doc_dict2[q_final_weight[j][0]]))
				if(len(doc_dict2[q_final_weight[j][0]])>0):
					lam2 = mew2/(mew2+len(doc_dict2[q_final_weight[j][0]]))
					bigm = lam2 * (doc_dict2[q_final_weight[j][0]].count(q1)/(len(doc_dict2[q_final_weight[j][0]])))+ (1-lam2)*unigm
				else:
					bigm = unigm
				#print (bigm)
				q_final_weight[j][1] = q_final_weight[j][1] + math.log(bigm)
			
		q_final_weight = sorted(q_final_weight,key=lambda x:int(x[1]),reverse=True)
		
		### writing first output to the file
		f=open(os.path.join(folder,"score_lm_bigram"), "a")
		
		for j in range (len(q_final_weight)):
			f.write(q_id+" Q0 "+str(q_final_weight[j][0])+" "+ str(j+1)+" "+str(q_final_weight[j][1])+" IndriQueryLikelihood\n")
		f.close()


if __name__ == "__main__":
	
	temp = True
	if(len(sys.argv)!=5):
		print("wrong command")
		print("command : lm_rerank [query-file] [top-100-file] [collection-file] [model=uni|bi]")
		temp = False
	if temp == True:
		if sys.argv[4] == "uni":
			function_unigram()
		elif sys.argv[4] == "bi":
			function_bigram()
		else:
			print("wrong command")
			print("command : lm_rerank [query-file] [top-100-file] [collection-file] [model=uni|bi]")


