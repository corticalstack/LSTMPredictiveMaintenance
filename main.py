import pandas as pd
from sklearn import preprocessing
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM

def load_data():
    return pd.read_csv('equipment_operations_logs.csv')


def data_discovery(df, tr):
    print('--- Shape')
    print('\tRow count:\t', '{}'.format(tr))
    print('\tColumn count:\t', '{}'.format(df.shape[1]))

    print('\n--- Row count by class')
    series = df['class'].value_counts()
    for idx, val in series.iteritems():
        print('\t{}: {} ({:6.3f}%)'.format(idx, val, ((val / tr) * 100)))

    print('\n--- Row count by id_machine')
    series = df['id_machine'].value_counts()
    for idx, val in series.iteritems():
        print('\t{}: {} ({:6.3f}%)'.format(idx, val, ((val / tr) * 100)))


def feature_engineering(df):
    df['cycle'] = df.groupby('id_machine').cumcount()

    # OneHotEncode class column with pandas
    df_onehotencode = pd.get_dummies(df['class'])
    df = pd.concat([df, df_onehotencode], axis=1)

    # Now remove class
    df.drop(['class'], axis=1, inplace=True)

    df_sum = df['Equipment Down'].iloc[::-1].cumsum()
    df['time_to_fail'] = df_sum.groupby(df_sum).cumcount()
    return df


def train_test_split(df):
    train_size = int(len(df) * 0.70)
    print('Training Observations: {}'.format(len(df[0:train_size])))
    print('Testing Observations: {}'.format(len(df[train_size:len(df)])))
    return df[0:train_size], df[train_size:len(df)]


def main():
    dataset = load_data()
    dataset = dataset.rename(columns={'id_source_primary_machine': 'id_machine', 'dt_ti_cycle_start': 'cycle_start',
                                      'tx_delay_class_description': 'class'})
    total_rows = dataset.shape[0]
    data_discovery(dataset, total_rows)

    dataset = feature_engineering(dataset)

    dataset.sort_values(by=['cycle_start', 'id_machine'], inplace=True)
    train, test = train_test_split(dataset)

    train['cycle_norm'] = train['cycle']
    cols_normalize = train.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset[['cycle', 'time_to_fail']] = min_max_scaler.fit_transform(dataset[['cycle', 'time_to_fail']])


    # pick a large window size of 50 cycles
    sequence_length = 50

    # pick a large window size of 50 cycles
    model = Sequential()
    model.add(LSTM(
        input_shape=(sequence_length, nb_features),
        units=100,
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        units=50,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=nb_out))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae', r2_keras])

    print(model.summary())

    # fit the network
    history = model.fit(seq_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0,
                                                          mode='min'),
                            keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True,
                                                            mode='min', verbose=0)]
                        )

    # list all data in history
    print(history.history.keys())
    print('finsihed')



if __name__ == '__main__':
    main()
