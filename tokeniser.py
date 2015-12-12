# tokenizing
# seperating by words or sentences

#DEFINITIONS:
    # corpora: a body of text eg : medical journals, presidential speeches,

    # lexicon - wods and their meanings
        # investor-speak .... regular english-speak
        # investor speak 'bull' : positive about the market
        # english speak 'bull' : an animal

from nltk.tokenize import sent_tokenize,word_tokenize

example_text = "Hello there ,what's up? I'm good. This weather is brilliant."

## tokemmise by sentences
print(sent_tokenize(example_text))
print(word_tokenize(example_text))


for i in word_tokenize(example_text):
    print i
