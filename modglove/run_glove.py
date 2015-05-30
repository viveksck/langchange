import subprocess

YEARS = range(2007,2008)

DATA_DIR = "/dfs/scratch0/google_ngrams/"
SAVE_FILE = DATA_DIR + "sglove-vecs-interesting/{year}-300vec"
INPUT_FILE = DATA_DIR + "modglove_train_interesting/{year}.shuf.bin"
INIT_FILE = DATA_DIR + "sglove-vecs-interesting/{year}-300vec.bin"
VOCAB_FILE = DATA_DIR + "modglove_train_interesting/{year}.vocab"

for year in YEARS:
    print "Running year", year
    subprocess.call(['./glove', 
            '-save-file', SAVE_FILE.format(year=year),
            '-threads', '70', 
            '-input-file', INPUT_FILE.format(year=year),
#            '-init-file', INIT_FILE.format(year=year-1),
            '-iter', '50', 
            '-vector-size', '300',
            '-binary', '2',
            '-vocab-file', VOCAB_FILE.format(year=year),
            '-verbose', '2'])
