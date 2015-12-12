import nltk
from nltk.corpus import state_union # presidential speeches
from nltk.tokenize import PunktSentenceTokenizer # a new tokenizer , which uses unsupervised machine learning, we can train it too on another data set

sample_text = state_union.raw("2006-GWBush.txt")
train_text = state_union.raw("2006-GWBush.txt")


custom_sent_tokenizer = PunktSentenceTokenizer(train_text) # Here we are traing the PunktSentenceTokenizer on this data set


tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            print tagged
    except Exception as e:
        print str(e)

process_content()

# forms a tuple with a word and a word origin (adjective,verb etc..)
