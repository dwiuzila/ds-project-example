# -*- coding: utf-8 -*-
import sys
sys.path.append('src')

import json
import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import mean_squared_log_error
from configs import config

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('models_output_filepath', type=click.Path())
@click.argument('reports_output_filepath', type=click.Path())
def main(input_filepath, models_output_filepath, reports_output_filepath):
    """ Train models and save them to "models" directory.
    """
    logger = logging.getLogger(__name__)
    logger.info('TRAIN MODEL')

    # load data
    X_train = load_npz(Path(input_filepath, 'X_train.npz'))
    X_valid = load_npz(Path(input_filepath, 'X_valid.npz'))
    Xb_train = load_npz(Path(input_filepath, 'Xb_train.npz'))
    Xb_valid = load_npz(Path(input_filepath, 'Xb_valid.npz'))
    y_train = np.load(Path(input_filepath, 'y_train.npy'))
    y_valid = np.load(Path(input_filepath, 'y_valid.npy'))
    logger.info('loading data')

    # load transformer
    y_scaler = joblib.load(Path(input_filepath, 'y_scaler.joblib'))
    logger.info('loading label transformer')

    # define datasets to be fit by the model: 2 for original, 2 for binary
    nets = 4
    models = [0] * nets
    xs_train = [Xb_train, X_train] * 2
    xs_valid = [Xb_valid, X_valid] * 2

    # define model (Functional API), fit, and save
    logger.info('training 4 neural networks...')
    for i in range(nets):
        train = xs_train[i]
        
        model_in = Input(shape=(train.shape[1],), dtype='float32', sparse=True)
        out = Dense(192, activation='relu')(model_in)
        out = Dense(64, activation='relu')(out)
        out = Dense(64, activation='relu')(out)
        out = Dense(1)(out)
        model = Model(model_in, out)
        model.compile(loss=config.loss, optimizer=Adam(learning_rate=config.learning_rate))
        
        for j in range(3):
            model.fit(
                x=train, y=y_scaler.transform(np.log1p(y_train.reshape(-1, 1))), 
                batch_size=2**(11 + j), epochs=1, verbose=1
            )
            
        
        model.save(Path(models_output_filepath, f'model {i+1}.h5'))
        models[i] = model
        logger.info(f'saving model {i+1}')

    # ensemble prediction on train dataset and log the error
    y_train_pred = np.mean([models[i].predict(xs_train[i])[:, 0] for i in range(nets)], axis=0)
    y_train_pred = np.expm1(y_scaler.inverse_transform(y_train_pred.reshape(-1, 1))[:, 0])
    train_score = mean_squared_log_error(y_train, y_train_pred, squared=False)
    logger.info('train RMSLE: {:.4f}'.format(train_score))

    # ensemble prediction on validation dataset and log the error
    y_valid_pred = np.mean([models[i].predict(xs_valid[i])[:, 0] for i in range(nets)], axis=0)
    y_valid_pred = np.expm1(y_scaler.inverse_transform(y_valid_pred.reshape(-1, 1))[:, 0])
    valid_score = mean_squared_log_error(y_valid, y_valid_pred, squared=False)
    logger.info('valid RMSLE: {:.4f}'.format(valid_score))

    # save results
    results = {'train_score': train_score, 'valid_score': valid_score}
    with open(Path(reports_output_filepath, 'model_score.json'), 'w') as f:
        json.dump(results, f)
    logger.info('saving reports')
    
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
