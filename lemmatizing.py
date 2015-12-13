## Lemmatizing ?
# Its a similiar operation to stemming , but the end result is an actual word, like a synonlyn

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer();

print(lemmatizer.lemmatize('',pos="a")); ## adjective
## default for lemmatizer is  pos="n" // a noun
## its better than stemming and will group a lot of words to one word ,so it's kind of more powerful
