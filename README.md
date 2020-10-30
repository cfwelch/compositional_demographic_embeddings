Coming Soon!

Working on cleaning and uploading code and instructions for:
- [ ] Data processing
- [ ] Attribute extraction
- [ ] Embedding training
- [ ] Model training
- [ ] Analysis

**Note**: We were not able to share data directly due to licensing issues. However, the data we downloaded is available [here](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) and we have scripts to perform the extraction the same way as described in our paper.

## Publication

More details of experiments run with this code can be found in the [our paper](https://arxiv.org/abs/2010.02986).

If you use this code please cite our paper:

```
@InProceedings{emnlp20compositional,
    title   = {Compositional Demographic Word Embeddings},
    author  = {Charles Welch and Jonathan K. Kummerfeld and Ver{\'o}nica P{\'e}rez-Rosas and Rada Mihalcea},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
    month = {November},
    year = {2020},
    location = {Online}
}
```

## Extracting Demographics of Reddit Users
If you want to sample and annotate new 'i am' posts you can use compose/sample_i_posts.py. The functions for extracting attributes from users are in compose/find_self_statements.py. Run this script to produce lists of users for each attribute. Replace DIR_LOC with the location of your Reddit data.

```
python find_self_statements.py --type gender
python find_self_statements.py --type location
python find_self_statements.py --type religion
python find_self_statements.py --type age
```

### Preprocessing
1. Put the speaker names for speakers of interest in top_speakers.
2. Run get_ts_posts.py -p -d all to get posts from this set of speakers from all years.
3. Run merge_ts_posts.py to combine these files and output author_json files in all_posts.
4. Run preprocess_all_posts.py to preprocess all_posts/author_json files.
5. Run sentence_tokenize.py to run CoreNLP tokenizer on all posts.