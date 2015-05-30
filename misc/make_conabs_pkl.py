import cPickle as pickle

infp = open("concreteness_ratings.csv")
infp.readline()
words = {}
for line in infp:
    line_info = line.split(",")
    words[line_info[0]] = float(line_info[2])

outfp = open("/dfs/scratch0/google_ngrams/stats/conabs-scores.pkl", "w")
pickle.dump(words, outfp)
