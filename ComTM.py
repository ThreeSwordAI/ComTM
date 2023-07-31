import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))
import re
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import wordnet
#import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from networkx.algorithms import community
import community as community_louvain
#from keybert import KeyBERT
#from summarizer import Summarizer,TransformerSummarizer


def unique(s):
    unique_list = []
    for x in s:
        if x not in unique_list:
            check = x.isnumeric()
            if(check):
                unique_list.append("num")
            else:
                unique_list.append(x)
            
    return unique_list

def news_articles_preprocess(art):
    pos = ["FW", "NN", "NNS", "NNP", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    wn = nltk.WordNetLemmatizer()
    articles = []
    for l in art:
        l = str(l)
        if(len(l)>300):
            article = l.split('\n\n')
            txt = article[0]
            txt2 = article[1:]
            bucket = []
            for i in txt2:
                tokenized = sent_tokenize(i)
                t = []
                for i in tokenized:
     
                    wordsList = nltk.word_tokenize(i)
                    wordsList = [w for w in wordsList if not w in stop_words]
                    tagged = nltk.pos_tag(wordsList)
                    for i in tagged:
                        t.append(i)
                s = ""
                for i in t:
                    if(i[1] in pos):
                        s=s+" "+i[0]
                    
                article = s
                article = article.lower()
                article = re.sub(r'[^\w\s]','',article)
                article = remove_stopwords(article)
                words_article = word_tokenize(article)
                wa = []
                for i in words_article:
                    bucket.append(wn.lemmatize(i))
            
            bucket= unique(bucket)
            
            
            tokenized = sent_tokenize(txt)
            t = []
            for i in tokenized:
     
                wordsList = nltk.word_tokenize(i)
                wordsList = [w for w in wordsList if not w in stop_words]
                tagged = nltk.pos_tag(wordsList)
                for i in tagged:
                    t.append(i)
            s = ""
            for i in t:
                if(i[1] in pos):
                    s=s+" "+i[0]
                    
            article = s
            article = article.lower()
            article = re.sub(r'[^\w\s]','',article)
            article = remove_stopwords(article)
            words_article = word_tokenize(article)
            wa = []
            for i in words_article:
                a = wn.lemmatize(i)
                if a in bucket:
                    wa.append(a)
            
            s = unique(wa)
            articles.append(s)
        else:
            articles.append([""])
    return articles

def scientific_abstracts_preprocess(art):
    pos = ["FW", "NN", "NNS", "NNP", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    wn = nltk.WordNetLemmatizer()
    articles = []
    for l in art:
        l = str(l)
        if(len(l)>300):
            txt = l
            tokenized = sent_tokenize(txt)
            t = []
            for i in tokenized:
                wordsList = nltk.word_tokenize(i)
                wordsList = [w for w in wordsList if not w in stop_words]
                tagged = nltk.pos_tag(wordsList)
                for i in tagged:
                    t.append(i)
            s = ""
            for i in t:
                if(i[1] in pos):
                    s=s+" "+i[0]
            article = s
            article = article.lower()
            article = re.sub(r'[^\w\s]','',article)
            article = remove_stopwords(article)
            words_article = word_tokenize(article)
            wa = []
            for i in words_article:
                a = wn.lemmatize(i)
                wa.append(a)
            
            s = unique(wa)
            articles.append(s)
            
            
    return articles


def wiki_data_preprocess(art, key):
    pos = ["FW", "NN", "NNS", "NNP", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    wn = nltk.WordNetLemmatizer()
    articles = []
    xy = 0
    for l in art:
        l = str(l)
        if(len(l)>300):
            txt = l
            tokenized = sent_tokenize(txt)
            t = []
            for i in tokenized:
                wordsList = nltk.word_tokenize(i)
                wordsList = [w for w in wordsList if not w in stop_words]
                tagged = nltk.pos_tag(wordsList)
                for i in tagged:
                    t.append(i)
            s = ""
            for i in t:
                if(i[1] in pos):
                    s=s+" "+i[0]
            article = s
            article = article.lower()
            article = re.sub(r'[^\w\s]','',article)
            article = remove_stopwords(article)
            words_article = word_tokenize(article)
            wa = []
            for i in words_article:
                a = wn.lemmatize(i)
                wa.append(a)
            wa = wa + key[xy]
            
            s = unique(wa)
            articles.append(s)
        xy+=1
            
            
    return articles







def comTM(topic_sen):
    counts = {}
    for i in topic_sen:
        for j in i:
            if j in counts:
                counts[j] +=1
            else:
                counts[j] = 1
    new_data = []
    for i in topic_sen:
        art=[]
        for j in i:
            if(counts[j]>1):
                art.append(j)
        new_data.append(art)
    c = 0
    for i in new_data:
        c = c + len(i)
        
    replace = {}
    w = []
    for i in new_data:
        for word in i:
            synonyms = []
            for syn in wordnet.synsets(word):
                for i in syn.lemmas():
                    synonyms.append(i.name())
            if(len(set(synonyms))==0):
                replace[word] = word
                w.append(word)
            else:
                for i in synonyms:
                    if i not in w:
                        replace[i] = word
                        w.append(word)    
                        
                        
    final_topic_sen = []
    for i in new_data:
        a = []
        for j in i:
            if(j in replace):
                a.append(replace[j])
            else:
                a.append(j)
        final_topic_sen.append(a)
        
    c = 0
    for i in final_topic_sen:
        c = c + len(i)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.80, min_df=2)
    documents = []
    for i in final_topic_sen:
        s = ""
        for j in i:
            s = s+" "+j
        documents.append(s)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    valid = tfidf_vectorizer.get_feature_names()
    final = []
    for i in final_topic_sen:
        s = []
        for j in i:
            if j in valid:
                s.append(j)
        if(len(s)>0):
            final.append(s)
            
    dt = []
    for i in final:
        s = ""
        for j in i:
            s = s+j+" "
        dt.append(s)
        
    cv = CountVectorizer(ngram_range=(1,1), stop_words = 'english')
    X = cv.fit_transform(dt)
    Xc = (X.T * X)
    Xc.setdiag(0)      
    
    names = cv.get_feature_names()
    df = pd.DataFrame(data = Xc.toarray(), columns = names, index = names)
    matrix = df.where(np.triu(np.ones(df.shape)).astype(np.bool))
    matrix = matrix.stack().reset_index()
    matrix.columns = ['Row','Column','Value']
    edge = matrix[matrix["Value"] != 0]
    node1 = edge.Row.values
    node2 = edge.Column.values
    value = edge.Value.values
    value = 1/value
    G = nx.Graph()
    for i in range(0, len(node1)):
        G.add_edge(node1[i], node2[i])
    partition = community_louvain.best_partition(G)
    
    m = max(list(partition.values()))
    total_com = m+1
    
    community_word = []
    for i in range(0, total_com):
        a=[]
        for j in partition:
            if partition[j] == i:
                a.append(j)
        community_word.append(a)
        
    weight = []
    for i in range(0, len(community_word)):
        #importent = {}
        #j_count = []
        com = community_word[i]
        G6 = nx.Graph()
        for e in range(0, len(node1)):
            if(node1[e] in com  and node2[e] in com):
                G6.add_edge(node1[e], node2[e], weight=value[e])
                
        centrality = nx.eigenvector_centrality(G6)      
        a = sorted(centrality.items())
        eg = {}
        for j in a:
            eg[j[0]] = j[1]
        sorted_dict = {k: v for k, v in sorted(eg.items(), key=lambda item: item[1],reverse=True)}
        weight.append(sorted_dict)
        
        aa = []
    for i in range(0, len(community_word)):
        #importent = {}
        #j_count = []
        com = community_word[i]
        G6 = nx.Graph()
        for e in range(0, len(node1)):
            if(node1[e] in com  and node2[e] in com):
                G6.add_edge(node1[e], node2[e], weight=value[e])
                
        centrality = nx.eigenvector_centrality(G6)      
        a = sorted(centrality.items())
        eg = {}
        for j in a:
            eg[j[0]] = j[1]
        b = sorted(eg, key=eg.get, reverse=True)[:10]
        aa.append(b)


    deg = []
    val = []
    for i in range(0, len(aa)):
        com = community_word[i]
        G = nx.Graph()
        for e in range(0, len(node1)):
            if(node1[e] in com  and node2[e] in com):
                G.add_edge(node1[e], node2[e], weight=value[e])
        c = {}
        v = []
        w = weight[i]
        for j in aa[i]:
            #degree_count = G.degree(j)
            #print(degree_count < 100)
            degree_count = 0
            for e in range(0, len(node1)):
                if(node1[e] == j  and node2[e] in com):
                    degree_count = degree_count + w[node2[e]]
                if(node2[e] == j  and node1[e] in com):
                    degree_count = degree_count + w[node1[e]]
                    
            c[j] = degree_count
            v.append(degree_count)
        deg.append(c)
        val.append(v)
        
        
    new_words = []
    for i in range(0, len(aa)):
        new = []
        minimum = min(val[i])
        w = weight[i]
        for j in range(0, len(aa)):
            if(i!=j):
                
                for k in aa[j]:
                    com = community_word[i]+[k]
                    G = nx.Graph()
                    b = False
                    for e in range(0, len(node1)):
                        if(node1[e] in com  and node2[e] in com):
                            G.add_edge(node1[e], node2[e], weight=value[e])
                            if(node1[e]==k or node2[e]==k):
                                b = True
                    if(b):
                        degree_count = 0
                        for e in range(0, len(node1)):
                            if(node1[e] == k  and node2[e] in com):
                                degree_count = degree_count + w[node2[e]]
                            if(node2[e] == k  and node1[e] in com):
                                degree_count = degree_count + w[node1[e]]
                    else:
                        degree_count = 0
                    
                    if minimum < degree_count:
                        #print(k,": ",degree_count)
                        new.append(k)
        new_words.append(new)
        
        
        bb = []
    for i in range(0, len(community_word)):
        #importent = {}
        #j_count = []
        com = community_word[i]+new_words[i]
        G6 = nx.Graph()
        for e in range(0, len(node1)):
            if(node1[e] in com  and node2[e] in com):
                G6.add_edge(node1[e], node2[e], weight=value[e])
                
        centrality = nx.eigenvector_centrality(G6)      
        a = sorted(centrality.items())
        eg = {}
        for j in a:
            eg[j[0]] = j[1]
        print("Topic Number: ", i+1,"------------>")
        b = sorted(eg, key=eg.get, reverse=True)[:10]
        print(b)
        bb.append(b)
        print()







def news_articles():
    df = pd.read_csv("corpus_news_articles.csv")
    article = df.article.values
    art = []
    for i in article:
        if(len(str(i))>0):
            art.append(str(i))
    article = art
    topic_sen = news_articles_preprocess(article)
    comTM(topic_sen)
    
    
def scientific_abstracts(): 
    df = pd.read_csv("corpus_scientific_abstracts.csv")
    article = df.article.values
    art = []
    for i in article:
        if(len(str(i))>0):
            art.append(str(i))
    article = art
    topic_sen = scientific_abstracts_preprocess(article)
    comTM(topic_sen)
  
 
    
def wiki_data():
    df = pd.read_csv("corpus_wiki_data.csv")
    """wiki_article = df.article.values
    art = []
    for i in wiki_article:
        if(len(str(i))>0):
            art.append(str(i))
        wiki_article = art
    

    for i in range(len(wiki_article)):
        index = wiki_article[i].find("\n\nSee also \n")
        if index != -1:
            wiki_article[i] = wiki_article[i][:index]
        index = wiki_article[i].find("\n\nSee also\n")
        if index != -1:
            wiki_article[i] = wiki_article[i][:index]
    
    
    summeries_list=[]
    bert_model = Summarizer()
    
    for i in range(0,len(wiki_article)):
        article_num=wiki_article[i]
        splitted_sum_add=""
        splitted_list=split_sentence(article_num)
        for j in splitted_list:
            bert_summary = ''.join(bert_model(j,min_length=40,max_length=100)) #, min_length=60
            splitted_sum_add=  splitted_sum_add+bert_summary #merging the sections of each article
        summeries_list.append(splitted_sum_add)
        
    keywords_list=[]
    kw_model = KeyBERT()
    
    for i in range(0,len(wiki_article)): 
        article_num=wiki_article[i]
        splitted_keys_add=[]
        splitted_list=split_sentence(article_num)
        for j in splitted_list:
            keywords = kw_model.extract_keywords(j) 
            a=kw_model.extract_keywords(j, keyphrase_ngram_range=(1, 3), stop_words="english",
                                  top_n=3, use_maxsum=True, diversity=0.7)
            splitted_keys_add.append(a) 
        result = union_2d_list(splitted_keys_add)
        keywords_list.append(result)
    
        
    
    
    topic_sen = wiki_data_preprocess(summeries_list, keywords_list)"""
    
    article = df.article.values
    
    art = []
    for i in article:
        if(len(str(i))>0):
            art.append(str(i))
    article = art
    len(article)
    key = df.keywords.values
    
    a = []
    for i in key:
        a.append(eval(i))
    
    key = a
    topic_sen = wiki_data_preprocess(article, key)
    comTM(topic_sen)

    
    

while(True):
    print("For News Articles, Press 1")
    print("For Scientific Abstracts, Press 2")
    print("For Wiki-Data, Press 3")
    
    t = int(input())
    
    if(t==1):
        news_articles()
        break
    if(t==2):
        scientific_abstracts()
        break
    if(t==3):
        wiki_data()
        break
    
    