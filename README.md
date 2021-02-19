Coming Soon!

Working on cleaning and uploading code and instructions for:
- [X] Data processing
- [X] Attribute extraction
- [ ] Embedding training
- [ ] Model training
- [ ] Analysis

**Note**: We were not able to share data directly due to licensing issues. However, the data we downloaded is available [here](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) and we have scripts to perform the extraction the same way as described in our paper. This repository contains generated JSON files containing fake data that the scripts can be tested on in the 'data' folder. These files contain fake authors with single letter names in `[a-z]`.

## Publication

More details of experiments run with this code can be found in the [our paper](https://arxiv.org/abs/2010.02986).

If you use this code please cite:

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

After running this for location, you will have files for locations in the demographic folder. You can then run `resolve_locations.py` from the `compose` folder to resolve locations to the following set, which was based on the amount of available data for each region:
1. USA
2. Asia
3. Oceania
4. United Kingdom
5. Europe
6. Africa
7. Mexico
8. South America
9. Canada

Next, run `complete_authors.py --find`, which will create a `complete_authors` file in the `demographic` folder, which will be used in the creation of matrix embeddings. There are a few important constants in `complete_authors.py` at the top of the file that can be changed:
* `MAX_AGE` and `MIN_AGE` are the range of accepted ages. If a user states that they are an age outside of this range, they will be excluded from the `complete_authors` list.
* `MAX_RANGE` is the largest accepted difference in stated ages. This acts as a simple heuristic to exclude users with incorrect ages. For instance, if a user states they are 20 in 2002 and 30 in 2003, this would not be possible. Our data spans 9 years, so we default this value to 9.
* `N_UNKS` is the criterion number of unknown demographic values that causes a user to be excluded. If you would like no users to be excluded, set this to the number of demographic variables plus one (4+1 for us). If you would like to exclude users who have no known demographic values, set this to the number of demographic variables (4 for us).
* `MIN_PERCENT` is the exclusion criteria for users who state both gender identities recognized by this study. For the less often expressed value, if it exceeds the `MIN_PERCENT`, the user is excluded.

## Preprocessing
1. Put the speaker names for speakers of interest in `top_speakers`. If you would like to run `find_bots_in_list.py` at this point, you can and it will output the names of speakers in your file that are known bots. At this point you can manually remove them if you'd like. A script to remove them can be run at step 6.
2. Run `get_ts_posts.py -p -d all` to get posts from this set of speakers from all years.
3. Run `merge_ts_posts.py` to combine these files and output author_json files in all_posts.
4. Run `preprocess_all_posts.py` to preprocess all_posts/author_json files.
5. Run `sentence_tokenize.py` to run CoreNLP tokenizer on all posts.
6. Run `rm_known_bots.py` to remove files in all_posts belonging to known bots.

## Creating Demographic Matrix Embeddings
The highest performing embeddings described in our paper use separate matricies for each demographic value and are learned using [Bamman et al.'s 2014 code](https://github.com/dbamman/geoSGLM). I have compiled separate JAR files and included separate config files for each demographic scenario.
1. First run `prepare_demographic_embed_data.py` to create the `java_ctx_embeds_reddit_demographic` file containing relevant data and the `reddit_vocab` file.
2. In the embeddings folder, run the shell script for each demographic (`./run_VARIABLE.sh`) to generate embeddings.

## Creating User Matrix Embeddings
1. TODO: Other paper...

## Create Word Category Plots
1. Follow the steps for creating embeddings for users above. The scripts for plotting category distributions are not currently available for demographic embeddings.
2. Run `vocab_counts.py` to get word frequencies per user.
3. Run `compare_spaces.py` to get word distances to generic embedding space per user.
4. Run `plot_cats_dist.py` to generate the graphs. Options are available for POS tags, LIWC, and ROGET categories. Figures can be saved or displayed with additional options, see `--help` for more details. (TODO test this)

## Other Scripts
* `token_counter.py` outputs a file called `token_counts` that contains counts of tokens from all speakers in your `top_speakers` file.
* `lexicon_map.py` is used by `plot_cat_dist.py` to graph LIWC and ROGET word class distributions and requires the `LIWC_PATH` and `ROGET_PATH` to be set in `lexicon_map.py` lines 9-10
* Running `complete_authors.py --plot` provides some useful plots of the demographic and post distributions of the dataset.