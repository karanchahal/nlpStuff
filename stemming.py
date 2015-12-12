## MAIN IDEA
## Getting to the root of the word, doing away with te fluff
## eg
# running -> run
# riding -> ride

# Algorithms
# Porter Stemmer
# Snowball Stemmer
# Lots of stemmers

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

#for w in example_words:
    #print(ps.stem(w))


## PRODUCTIVITY HACK !
stemmed = [ps.stem(w) for w in example_words]
