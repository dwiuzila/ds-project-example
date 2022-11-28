# -*- coding: utf-8 -*-
import sys
sys.path.append('src')

import pandas as pd
from sklearn.model_selection import KFold
from configs import config

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Split dataset into train and validation.
    """
    logger = logging.getLogger(__name__)
    logger.info('SPLIT DATASET')

    # read train dataset
    logger.info('reading raw data')
    train = pd.read_csv(Path(input_filepath, 'train.tsv'), sep='\t')
    train = train[train['price'] > 0].reset_index(drop=True)

    # split train dataset into train and validation dataset
    cv = KFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train))
    train, valid = train.iloc[train_ids], train.iloc[valid_ids]
    logger.info(f'train shape after split: {train.shape}')
    logger.info(f'valid shape after split: {valid.shape}')

    # save split
    logger.info('saving train and validation split')
    train.to_csv(Path(output_filepath, 'train.csv'), index=False)
    valid.to_csv(Path(output_filepath, 'valid.csv'), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
