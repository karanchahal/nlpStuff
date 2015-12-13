# What is chunking?

# you might have mnay nouns in a sentence ,that is we have a sentence and there are a lot of things in a sentence
# no nessesarily talking about eh same topic

## putting same code as speech tagging


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

            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?} """ #RB ,RBR ,RBS adverbs, . is any character except for newline,  so RB.? any form of an adverb
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            #print chunked
            chunked.draw(); # for humans

    except Exception as e:
        print str(e)

process_content()

# forms a tuple with a word and a word origin (adjective,verb etc..)
## end of speechtagging code
