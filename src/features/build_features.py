# -*- coding: utf-8 -*-
import sys
sys.path.append('src')

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from operator import itemgetter
from configs import config

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn split data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('BUILD FEATURES')

    # read train and validation dataset
    logger.info('reading train and validation dataset')
    train = pd.read_csv(Path(input_filepath, 'train.csv'))
    valid = pd.read_csv(Path(input_filepath, 'valid.csv'))

    # define preprocessing functions
    def preprocess(df):
        df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
        df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
        return df[['name', 'text', 'shipping', 'item_condition_id']]

    def on_field(f, *vec):
        return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

    def to_records(df):
        return df.to_dict(orient='records')

    # define vectorizing function
    vectorizer = make_union(
        on_field('name', Tfidf(max_features=config.max_features, token_pattern='\w+')),
        on_field('text', Tfidf(max_features=config.max_features, token_pattern='\w+', ngram_range=(1, 2))),
        on_field(['shipping', 'item_condition_id'],
                FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=-1)

    # preprocess and vectorize both train and validation dataset
    X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
    X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
    logger.info(f'X_train shape after preprocessed: {X_train.shape} of {X_train.dtype}')
    logger.info(f'X_valid shape after preprocessed: {X_valid.shape} of {X_valid.dtype}')

    # log-transform target variable
    logger.info('log-transforming target variable')
    y_scaler = StandardScaler()
    y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    y_train = train['price']
    y_valid = valid['price']

    # delete raw train and validation dataset to save memory
    del train
    del valid

    # make binary datasets for both train and validation
    logger.info('making binary datasets for both train and validation')
    Xb_train, Xb_valid = [x.astype(bool).astype(np.float32) for x in [X_train, X_valid]]

    # save processed dataset
    X_train_path = Path(output_filepath, 'X_train.npz')
    X_valid_path = Path(output_filepath, 'X_valid.npz')
    Xb_train_path = Path(output_filepath, 'Xb_train.npz')
    Xb_valid_path = Path(output_filepath, 'Xb_valid.npz')
    y_train_path = Path(output_filepath, 'y_train.npy')
    y_valid_path = Path(output_filepath, 'y_valid.npy')
    save_npz(X_train_path, X_train)
    save_npz(X_valid_path, X_valid)
    save_npz(Xb_train_path, Xb_train)
    save_npz(Xb_valid_path, Xb_valid)
    np.save(y_train_path, y_train)
    np.save(y_valid_path, y_valid)

    # save transformer
    y_scaler_path = Path(output_filepath, 'y_scaler.joblib')
    joblib.dump(y_scaler, y_scaler_path)

    logger.info('saving data and transformer')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
