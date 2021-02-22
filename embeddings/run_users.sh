# input file
DATA=../sts_combined_users

# vocab file
VOCABFILE=../reddit_user_vocab

# set of metadata facets
FEATUREFILE=../top_speakers

# output file to write embeddings to
OUTFILE=data/embed_test_users

# max vocab size
MAXVOCAB=1000000

# dimensionality of embeddings
DIMENSIONALITY=100

# L2 regularization parameter
REGULARIZE=True
L2=0.0001

# Number of epochs and threads to run
EPOCHS=1
THREADS=2

./runjava_users geosglm.ark.cs.cmu.edu/GeoSGLM $DATA $VOCABFILE $FEATUREFILE $OUTFILE $MAXVOCAB $DIMENSIONALITY $L2 $REGULARIZE $EPOCHS $THREADS
