## what is a corpora, accessing A corpora
import nltk

#print nltk.__file__; ## we do this to get the location of our packages in the system of nltk library


from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw('bible-kjv.txt')
tok = sent_tokenize(sample)
print(tok[5:15]);

## Dont' generally load raw text cause some files are stored as .txt but they are
## tables etc.
