import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

l = str(input("Enter your sentence : "))

doc = nlp(l)
if doc == []:
    print("Sorry, the model could not find any named entity in the entered sentence.")
print([(X.text, X.label_) for X in doc.ents])