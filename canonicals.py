import re
import pymorphy2
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


morph = pymorphy2.MorphAnalyzer()
wordnet_lemmatizer = WordNetLemmatizer()

    
def getWords(text, stop_words=[]):
    regexp = '[a-zA-Zа-яА-Я]+'
    
    return filter(lambda x: x not in stop_words and len(x) > 1, re.findall(regexp, text.lower()))

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
        
def _word2canonical4w2v(word):
    elems = morph.parse(word)
    my_tag = ''
    res = []
    for elem in elems:
        if 'VERB' in elem.tag or 'GRND' in elem.tag or 'INFN' in elem.tag:
            my_tag = 'V'
        elif 'NOUN' in elem.tag:
            my_tag = 'S'
            
        normalised = elem.normalized.word
        
        if my_tag == '':
            res.append((normalised, ''))
            
        res.append((normalised, my_tag))
        
    tmp = list(filter(lambda x: x[1] != '', res))
    if len(tmp) > 0:
        return tmp[0]
    else:
        return res[0]

def word2canonical(word, lang='rus'):
    if lang == 'rus':
        return _word2canonical4w2v(word)[0]
        
    elif lang == 'eng':
        word, tag = nltk.pos_tag([word], lang='eng')[0]
        wordnet_tag = get_wordnet_pos(tag)
        try:
            lemmatized_word = wordnet_lemmatizer.lemmatize(word, wordnet_tag)
        except KeyError:
            lemmatized_word  = wordnet_lemmatizer.lemmatize(word)
                
    return lemmatized_word
        
def text2canonicals(text, stop_words=[], lang='rus'):
    words = []
    for word in getWords(text, stop_words=stop_words):
        words.append(word2canonical(word.lower(), lang))

    return words