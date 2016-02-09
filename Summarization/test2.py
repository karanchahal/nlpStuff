from nltk import word_tokenize

dic = {1:[]}
sent = "Jack is a dull boy"

tokens = word_tokenize(sent)
dic[1] = tokens;
if 1 in dic:
    print(dic[1])
    print("hello")