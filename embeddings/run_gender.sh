# input file
DATA=../java_ctx_embeds_reddit_demographic

# vocab file
VOCABFILE=../reddit_vocab

# set of metadata facets
FEATUREFILE=data/reddit_demographic_single_gender

FEATURETYPE=gender

# output file to write embeddings to
OUTFILE=data/embed_test_gender

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

./runjava geosglm.ark.cs.cmu.edu/GeoSGLM $DATA $VOCABFILE $FEATUREFILE $OUTFILE $MAXVOCAB $DIMENSIONALITY $L2 $REGULARIZE $EPOCHS $THREADS $FEATURETYPE
