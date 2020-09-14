# Usage

Get the [ALF200k dataset](https://github.com/dbis-uibk/ALF200k) in a Mongo DB.
Export the needed files as specified in `tools/create_dataset_lfm.py`.
Additionally, crawl raw lyrics and then run `tools/create_dataset_lfm.py`
followed by `tools/extract_genre_lfm.py` to store the dataset in
`data/processed/dataset-lfm-genres.pickle`.
