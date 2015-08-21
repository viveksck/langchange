from nltk.corpus import brown
from nltk.tag import UnigramTagger
import cPickle as pickle

VERBS = set(['VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP'])
NOUNS = set(['NN', 'NNS'])
ADJECTIVES = set(['JJ', 'JJS', 'JJT', 'JJS'])
ADVERBS = set(['RB'])

INPUT_FILE = "/dfs/scratch0/googlengrams/2012-eng-fic/info/commonnonstop-1900-2000-8-6.pkl"

def write_word_list(filename, word_list):
    out_fp = open(filename, "w")
    print >> out_fp, "\n".join(word_list)

if __name__ == '__main__':
    in_fp = open(INPUT_FILE, "rb") 
    words = pickle.load(in_fp)
    tagger = UnigramTagger(brown.tagged_sents())
    verb_words = []
    noun_words = []
    adj_words = []
    adv_words = []
    for word in words:
        tag = tagger.tag([word])[0][1]
        if tag in VERBS:
            verb_words.append(word)
        elif tag in NOUNS:
            noun_words.append(word)
        elif tag in ADJECTIVES:
            adj_words.append(word)
        elif tag in ADVERBS:
            adv_words.append(word)
    write_word_list("verbs.txt", verb_words)
    write_word_list("nouns.txt", noun_words)
    write_word_list("adjs.txt", adj_words)
    write_word_list("advs.txt", adv_words)
