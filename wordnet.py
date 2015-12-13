from nltk.corpus import wordnet

syns = wordnet.synsets("program");

#print syns[0];
# synset
print syns[0].name()

#print syns[0].lemmas() ## to get lemmas


#just the word
print syns[0].lemmas()[0].name() ## get a name

# DEFINITIONS
print syns[0].definition()

## example

print syns[0].examples()


## Getting synonyms and antonyms


synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        #print l
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())


#print set(synonyms);
#print set(antonyms);


## SEMANTIC SIMILIARITY

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

## we can compare semantic similiarity between words

print (w1.wup_similarity(w2))

## MOre examples
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")

print (w1.wup_similarity(w2))


w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")


print (w1.wup_similarity(w2))
