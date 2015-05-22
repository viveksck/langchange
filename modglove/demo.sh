#!/bin/bash

CORPUS=text8
VOCAB_FILE=vocab.txt
VOCAB_FILE=/dfs/scratch0/google_ngrams/modglove_train_interesting/2005.vocab
COOCCURRENCE_FILE=cooccurrence.shuf.bin
COOCCURRENCE_SHUF_FILE=/dfs/scratch0/google_ngrams/modglove_train_interesting/2005.bin
INIT_FILE=/dfs/scratch0/google_ngrams/
SAVE_FILE=vectors2
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=1
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=1
X_MAX=10

./glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
