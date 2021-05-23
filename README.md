1. Probabilistic Retrieval Reranking -

python3 prob_rerank.py [query-file] [top-100-file] [collection-file] [expansion-limit]

When we run the prob_rerank.py file it will create an "output" folder in the current working directory and will save the output in that folder. "score_1" is the file we get after expanding the query by one term similarly with other files.

2.Language Model -

python3 lm_rerank.py [query-file] [top-100-file] [collection-file] [model=uni|bi]

When we run the lm_rerank.py it will create an "output_lm_uni" folder for model = uni and store the output file in that folder  and form model = bi it will create an "output_ln_bi" folder and store the output file in that folder.