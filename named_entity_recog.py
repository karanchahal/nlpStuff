
# what is named entity recognition
# its basically chunking bu using named entities?

## Named entities chunks data by name entities, person etc organisation, money etc
import nltk
from nltk.corpus import state_union # presidential speeches
from nltk.tokenize import PunktSentenceTokenizer # a new tokenizer , which uses unsupervised machine learning, we can train it too on another data set

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)


            namedEnt = nltk.ne_chunk(tagged)
            namedEnt.draw(); # for humans

    except Exception as e:
        print str(e)




sample_text = state_union.raw("2006-GWBush.txt")
train_text = state_union.raw("2006-GWBush.txt")


custom_sent_tokenizer = PunktSentenceTokenizer(train_text) # Here we are traing the PunktSentenceTokenizer on this data set


tokenized = custom_sent_tokenizer.tokenize(sample_text)
process_content();
