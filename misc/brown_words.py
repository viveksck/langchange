from nltk.corpus import brown
from nltk.tag import UnigramTagger
import cPickle as pickle

INPUT_FILE = "/dfs/scratch0/googlengrams/2012-eng-fic/info/commonnonstop-1900-2000-8-6.pkl"

def write_word_list(filename, word_list):
    out_fp = open(filename, "w")
    print >> out_fp, "\n".join(word_list)

if __name__ == '__main__':
    in_fp = open(INPUT_FILE, "rb") 
    words = pickle.load(in_fp)
    tagger = UnigramTagger(brown.tagged_sents())
    good_words = []
    for word in words:
        tag = tagger.tag([word])[0][1]
        if tag == None:
            continue
        if "NP" in tag:
            continue
        good_words.append(word)
    write_word_list("brown.txt", good_words)
