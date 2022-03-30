from model import Predictions
import spacy
from spellcorrector import SpellCorrector, BhuvanaSpellCorrector

words = Predictions( r'C:\Users\gaura\OneDrive\Desktop\Handwriting_Classification\Images')
print(words.result)
newline = SpellCorrector(words.result[0])
print(newline.result)
