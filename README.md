# ComTM

## To run the ComTM pipeline:
1/ Run ComTM.py file in Python 3 environment

2/ After running ComTM, the program will ask for input

	where 1 is for News Articles
	where 2 is for Scientific Abstracts
	where 3 is for Wiki Data
 
3/ The program will read .csv file

	where 1 is for corpus_news_articles.csv
	where 2 is for corpus_scientific_abstracts.csv
	where 3 is for corpus_wiki_data.csv
 
4/ In the .csv file, the data should be in the column name "article". For wiki_data there also should be a column name "keywords". 


Required Libraries: numpy, pandas, nltk, re, gensim, sklearn, networkx, community

## If you are using ComTM, please refer to the following paper:

[Topic Modeling Using Community Detection on a Word Association Graph](https://aclanthology.org/2023.ranlp-1.98/)
