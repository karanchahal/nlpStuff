from __future__ import division, unicode_literals
import nltk
import itertools
import string
import operator
from operator import itemgetter
import os
import sys
import numpy as np
from sklearn.cluster import spectral_clustering
from itertools import product
import math
import time
import pickle
import matplotlib.pyplot as plt
from pulp import *

count_dp = 0;

""" This class generates objects which store the following sentence characteristics:
    Index: The order in which sentences are stored in the document
    Score: The Partial Entailment Score alloted to each sentence. The formula was = (sigma(score of all sentences to our object sentence)/(number of fragments of that sentence))
    Text: Textual Content of the sentence
"""

class Rank_Object:
    index = -1;
    score =0;
    text =""
    def __init__(self,index):
        self.index = index;

""" This class generates objects which store the following fragment characteristics:
    Index: The order in which fragment are stored in the document
    Text: Textual Content of the sentence
"""

class Frag_Object:
    index = -1;
    serial = -1;
    text ="";
    def __init__(self,index,serial):
        self.index = index;
        self.serial = serial;

""" This removes all the sentence elements apart from that inside quotes ans stores it into a seperate fragment.If there
    are no quotes in the sentence we leave the sentence alone
"""
def quotes(s):
    if '"' not in s and '``' not in s and '\'\'' not in s :
        return s
    n = "";
    found=0;
    x = s.split(' ');
    for i in x :
        if('"' in i or '``' in i or '\'\'' in i):
            if found==0 :
                n = n +i+' ';
                found=1;
                continue;
            else:
                n=n+i+' ';
                break;
            if found==1 :
                n=n+i+' ';
    return n

"""  The WMVC function """
def wmvc(matrix,n,sentenceTokens,fileName):
    print("inside wmvc")
    #threshold
    m=np.mean(matrix)

    #variable for ilp
    values = [None for _ in range(n)]

    #initialize ilp variables
    for i in range(n):
       values[i] = LpVariable("x"+str(i),0,1,'Binary')
    prob = LpProblem("problem",LpMinimize)
    #print "Calculate ConnScore for every vertex"
    #calculate ConnScore for every vertex
    arc = [0 for i in range(n)]
    for i in range(n):
        for j in range(n):
            arc[i] = arc[i] + matrix[i][j]
    for i in range(n):
        arc[i] *= -1;
    large_positive_constant = abs(min(arc)) + 1
    for i in range(n):
            arc[i] = arc[i] + large_positive_constant

    #add constrains to the ilp problem
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > m or matrix[j][i] > m:
                prob += values[i] + values[j] >= 1
    #print "function to be minimized"
    #function to be minimized
    z = arc[0] * values[0]
    for i in range(1,n):
        z += arc[i] * values[i]
    prob += z
    #print "solve ilp and get result"
    #solve ilp and get result
    status = prob.solve(GLPK(msg=0))
    str1 = [value(values[i]) for i in range(n)]
    #print(str(str1))

    #length of individual sentences
    leng = [None for _ in range(n)]
    for i in range(n):
        leng[i]= len(sentenceTokens[i][0].split(' '));

    print("Generating summaries to " + 'Sum/'+fileName)
    result = open("Sum/"+folder+"/summary.txt", 'w')

    k=0;
    words=0;
    for i in range(n):
        if words >= 100 :
            break
        if value(values[i])>=0.5 :
            words=words+leng[i]
            k=k+1;
            result.write(sentenceTokens[i][0]+'\n')

    result.close()
    print("length of input = " + str(len(sentenceTokens)) + "; length of summary : " + str(k) + "\n------\n\n")

""" Finds out how many times word exists in the given sentence tokens"""
def n_containing(word, sentenceTokens):
    print("in n_containing")
    return sum(1 for sentence in sentenceTokens  if word in sentence)

def generate_score(wordTokens1,wordTokens2,n):

    idf_Ti=0.000000000000001
    idf_Tj=0.000000000000001
    idf_comm=0.0

    for i in wordTokens1:
                idf1= math.log(n/float(1+n_containing(i,li)))
                idf_Ti+=idf1
                if i in wordTokens2:
                    idf_comm += idf1
    #Assigning values and calculte max,min
    return idf_comm/float(idf_Ti)

def generate_sentence_list(fragment_features,no_of_frags):
    ranklist = [];

    for i in range(len(sentences)):
        redundant = 0;
        for j in sentences[i]:
            if j=='"' or j == "'" or j == '`':
                redundant = 1
                break

        if(redundant == 1):
            continue

        obj= Rank_Object(i);
        obj.text = sentences[i]
        ranklist.append(obj);
        ranklist[-1].score = 0
        for j in range(len(frags)):
            ranklist[-1].score += fragment_features[j][i]/len(frags)


    ranklist.sort(key = operator.attrgetter('score'),reverse=True);
    return ranklist

def remove_redundant_sentences(ranklist,sim_matrix):
    finallist = [];
    finallist.append(ranklist[0]);

    for i in range(len(ranklist)):
        if(ranklist[i].score != finallist[-1].score):

            #check for the similiarity score bar
           flag =0;
           for j in finallist:
            if(sim_matrix[j.index][ranklist[i].index] > 0.9):
                    flag = 1;
                    break;

           if(flag == 0):
            finallist.append(ranklist[i]);

    return finallist

def generate_new_sim_matrix(finallist,sim_matrix):
    new_sim_matrix = np.zeros(shape=(len(finallist),len(finallist)))
    for i in range(len(finallist)):
        for j in  range(len(finallist)):
            new_sim_matrix[i][j] = sim_matrix[finallist[i].index][finallist[j].index]

    return new_sim_matrix


def generate_cluster_matrix(finallist,labels):
    cluster_matrix = []

    for i in range(15):
        cluster_matrix.append([])

    for i in range(len(finallist)):
        cluster_matrix[labels[i]].append(finallist[i])

    return cluster_matrix

def generate_wmc_matrix_and_sent_tokens(cluster_matrix,sim_matrix):
    i =0
    size = 0
    finallist = []
    sentence_tokens = []
    for cluster in cluster_matrix:
        print(i)
        finallist.append(cluster[0])
        #print cluster[0].text
        temp_tokens = sent_detector.tokenize(cluster[0].text.strip())
        sentence_tokens.append(temp_tokens)
        i += 1
    """Getting d matrix and sentence tokens list for WMVC  """
    m = 15
    #print len(finallist)
    wmc_matrix = np.zeros(shape=(m,m))
    for i in range(len(finallist)):
        for j in range(len(finallist)):
            wmc_matrix[i][j] = sim_matrix[finallist[i].index][finallist[j].index]
    return wmc_matrix,sentence_tokens

""" Extracts fragments , gives a PE score """
def extractSentences(text,folder):

    n = len(frags)
    m = len(sentences)
    li =[]
    no_of_frags = [];
    for i in range(m):
        no_of_frags.append(0);

    for i in frags:
        no_of_frags[i.index] += 1;
        li.append(i.text);

    sim_matrix =np.zeros(shape=(m,m))
    for i in range(m):
        for j in range(m):
            sim_matrix[i][j]=0

    d = [[None for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            d[i][j]=0
    x=0;y=0;


    dic = {}
    count_dp = 0;
    not_dp = 0;
    for first in sentences:
        wordTokens1=nltk.word_tokenize(first)
        for second in frags:

            if second.serial in dic:
                 wordTokens2 = dic[second.serial]
                 count_dp = count_dp + 1;
            else:
                 wordTokens2=nltk.word_tokenize(second.text)
                 dic[second.serial] = wordTokens2
                 not_dp = not_dp + 1

            d[y][x]=generate_score(wordTokens1,wordTokens2,n)
            if(second.index != x):
                if(no_of_frags[second.index] != 0):
                    sim_matrix[second.index][x] += d[y][x]/(no_of_frags[second.index])
            y=y+1
        y=0;
        x=x+1;

    print("Count of dp hit is : " + str(count_dp))
    print("Count of no hit is : " + str(not_dp))

    ranklist = generate_sentence_list(d,no_of_frags)
    finallist = remove_redundant_sentences(ranklist,sim_matrix)


    print("Generating summaries to " + 'Sum/')
    result = open("Sum/"+folder+"/summary.txt", 'w')

    k=0;
    words=0;
    for i in range(len(finallist)):
        result.write(finallist[i].text + '\n');
        words += len(nltk.word_tokenize(finallist[i].text))
        if words >= 100:
            break
    result.close;


    new_sim_matrix = generate_new_sim_matrix(finallist,sim_matrix);
    labels = spectral_clustering(new_sim_matrix,n_clusters = 15,eigen_solver='arpack',random_state=1)

    print (labels)
    fraglist = []
    for r in finallist:
        fraglist.append(no_of_frags[r.index])


    cluster_matrix = generate_cluster_matrix(finallist,labels)
    wmc_matrix,sentence_tokens = generate_wmc_matrix_and_sent_tokens(cluster_matrix,sim_matrix)


    wmvc(wmc_matrix,15,sentence_tokens,"Sum")





start_time=time.time()
if not os.path.exists("Sum"):
	os.mkdir("Sum")
text = ""
#retrieve each of the articles
folders=os.listdir("Orig")
if not os.path.exists("TE"):
	os.mkdir("TE")

k = 0
count_dp = 0
for folder in folders:
    if not os.path.exists("Sum/"+folder):
	    os.mkdir("Sum/"+folder)
    frags=[]
    sentences=[]
    articles = os.listdir("Orig/"+folder)
    for article in articles:
        print('Reading articles/' + article)
        articleFile = open('Orig/'+folder+'/'+article, 'r')
        text += articleFile.read()

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    frags = sent_detector.tokenize(text)
    sentences = sent_detector.tokenize(text)
    NFrags = []
    j =0; # setting iterator for sentences
    for i in frags:
        new = i
        new = quotes(new)
        #fragment them, remove these words.
        new = new.replace(" and ","***")
        new = new.replace(" but","***")
        new = new.replace(", ","***")
        new = new.replace(" either ","***")
        new = new.replace(" or ","***")
        new = new.replace(" neither ","***")
        new = new.replace(" nor ","***")
        new = new.replace(" yet ","***")
        new = new.replace(";","***")
        new = new.replace(":","***")
        new = new.replace(" as far as i know ","***")
        new = new.replace("frankly speaking","***")
        new = new.replace(" so ","***")
        new = new.replace("for example","***")
        new = new.replace("for instance","***")
        new = new.replace("as an example","***")
        new = new.replace("to illustrate","***")
        new = new.replace("as an illustration","***")
        new = new.replace("not only","***")
        new = new.replace("moreover","***")
        new = new.replace("furthermore","***")
        new = new.replace("in addition to","***")
        new = new.replace("in addition","***")
        new = new.replace("likewise","***")
        new = new.replace("similarly","***")
        new = new.replace("as well as","***")
        new = new.replace("most probably","***")
        new = new.replace("just in case","***")
        new = new.replace("as soon as possible","***")
        new = new.replace("on the other hand","***")
        new = new.replace("in contrast to","***")
        new = new.replace(" as much as ","***")
        new = new.replace(" nevertheless","***")
        new = new.replace(" even so","***")
        new = new.replace(" even though","***")
        new = new.replace(" although","***")
        new = new.replace(" despite","***")
        new = new.replace(" as a result","***")
        new = new.replace(" therefore","***")
        new = new.replace(" thus","***")
        new = new.replace(" as a consequence","***")
        new = new.replace(" consequently","***")
        new = new.replace(" in conclusion","***")
        new = new.replace(" in summary","***")
        new = new.replace(" finally","***")
        new = new.replace(" meanwhile","***")
        new = new.replace(" whereas","***")

        ##breaking before
        new = new.replace(" such as","*** such as")
        new = new.replace(" namely","*** namely")
        new = new.replace(" specifically","*** specifically")
        new = new.replace(" when","*** when")
        new = new.replace(" while","*** while")
        new = new.replace(" which","*** which")
        new = new.replace(" who","*** who")
        new = new.replace(" whose","*** whose")
        new = new.replace(" where","*** where")
        new = new.replace(" whose","*** whose")
        new = new.replace(" because","*** because")
        new = new.replace(" even if","*** even if")
        new = new.replace(" as if","*** as if")
        new = new.replace(" as long as","*** as long as")
        new = new.replace(" now that","*** now that")
        new = new.replace(" rather than","*** rather than")
        new = new.replace(" whenever","*** whenever")
        new = new.replace(" while","*** while")

        new = new.replace("(","***(")
        new = new.replace(")",")***")

        li =[]
        li = new.split('***');
        for i in li:
            obj = Frag_Object(j,k);
            obj.text = i;
            NFrags.append(obj);
            k = k +1;

        j += 1

    #print sentences

    i = 0
    frags = NFrags[:]
    while(i < len(NFrags)):
        #remove empty words to count the number of words
        words = NFrags[i].text.split(' ')
        no = words.count("")
        for j in range(0,no):
            words.remove("")
        no = words.count(" ")
        for j in range(0,no):
            words.remove(" ")

        #remove small fragments
        if(len(words) <= 3):
            frags.remove(NFrags[i]) #remove fragments having ALL words starting with a capital letter
        else:
            flag = 0
            for word in words:
                if word[0] > 'Z' or word[0] < 'A':
                    flag += 1

            if flag <= 1:
                frags.remove(NFrags[i])
        i += 1


    extractSentences(text,folder)
print("...%s seconds..." %(time.time()-start_time))
print(count_dp)