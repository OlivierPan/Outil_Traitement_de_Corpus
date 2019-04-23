# coding:utf-8
# Etudiant, Pan Wenzhu, Paris Nanterre, Master TAL
import spacy
from spacy.util import minibatch, compounding
import random
# Pour le devoir au choix, j'ai pris le deuxième devoir qui consiste à update un modèle spacy sur la reconnaissance des entités nommées.
# J'ai choisi de travailler sur deux chansons de rap, l'une est <Chino> de Ninho, l'autre <Dans le mia> de IAM.


nlp=spacy.load('fr_core_news_sm')

# Quand on lance le spacy sur les paroles, on trouve des erreurs ci-dessous.
"""
[('Chino', 'MISC'), ('Ninho', 'MISC')]
[('Mes habits', 'MISC')]
[('Mes habits', 'MISC')]
[('Mes habits', 'MISC')]
[('Numéro 9', 'MISC')]
[('Bac+8', 'LOC')]
[('Fendi', 'PER'), ('Umbro', 'LOC')]
[('Amnézia', 'LOC')]
[('Danser le Mia', 'PER'), ('IAM', 'ORG')]
[('Stan Smith', 'PER')]
[('Ray Ban', 'PER'), ('Tacchini', 'PER')]
[('Nebuloni', 'PER')]
[('Cameo Midnight Star', 'PER')]
[('Shalamar', 'LOC')]
[('Marseille', 'LOC')]
"""

# On prépare le train-data avec l'annotaion de l'index des entités et leur lable 
TRAIN_DATA=[
	("Mes habits sont chers comme mes amis",{"entities":[]}),
	("Le daron voulait la fac de droit, le Bac+8",{"entities":[(37,42,"MISC")]}),
	("J'ai sorti le Fendi rangé le Umbro et j'ai des cartouches en cas d'embrouille",{"entities":[(14,19,"PRODUCT"),(29,34,"PRODUCT")]}),
	("Amnézia pour m'apaiser on m'a conseillé de me mettre au vert",{"entities":[]}),
	("Paroles de Danser le Mia par IAM",{"entities":[(11,24,"WORK_OF_ART"),(29,32,"ORG")]}),
	("Dès qu'ils passaient Cameo Midnight Star",{"entities":[(21,26,"WORK_OF_ART"),(27,40,"WORK_OF_ART")]}),
	("SOS Band Delegation ou Shalamar",{"entities":[(0,8,"ORG"),(9,19,"ORG"),(23,31,"ORG")]}),
]

# On passe le train-data à minibathc de Spacy

ner = nlp.get_pipe("ner")
# add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(100):
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(
                texts,  # batch of texts
                annotations,  # batch of annotations
                drop=0.5,  # dropout - make it harder to memorise data
                losses=losses,
            )
        print("Losses", losses)

# test the trained model
for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
   
print("---------------&&&&&&&&&&&&&&&&&&&&&&-----------------")

# On charge le nouvel modèle après l'entraînement et on teste

nlp.to_disk('/users/pansa/desktop/french_rap')
my_nlp=spacy.load('/users/pansa/desktop/french_rap')
for text, _ in TRAIN_DATA:
    doc = my_nlp(text)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])



txt=["Bac+8 est un diplome util","Parmi les groupes de rapeurs, j'aime le mieux IAM."]
toto=list(nlp.pipe(txt))
for to in toto:
    print([(tok.text,tok.pos_,tok.lemma_) for tok in to])
    if to.ents:
        print([(ent.text,ent.label_) for ent in to.ents])
        
# Conclusion: après l'entraînement, il reste encore des erreurs mais déjà moins.
