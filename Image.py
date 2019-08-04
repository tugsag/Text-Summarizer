from PIL import Image
import pytesseract as pt
import nltk, re

def read_image(file):
    text=pt.image_to_string(Image.open(file))
    ##re.sub('\n', ' ', text)
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        sent.strip()
    return ' '.join(sentences)
