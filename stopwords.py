## THIS SCRIPT REMOVES STOP WORDS FROM A SENTENCE

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_text = "This is an example showing stopword filteration and I am loving it."

stop_words = set(stopwords.words("english"))

#for i in stop_words: # gets pythons nltk stopwords list
    #print i

# We  can also append a stopword

words = word_tokenize(example_text)

filtered_sentence = []

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)

#print filtered_sentence;


## NOW FOR THE PRODUCTIVITY HACKS,A ONE LINE TO DO ALL THIS

filtered_sentence = [w for w in words if not w in stop_words]

print filtered_sentence
