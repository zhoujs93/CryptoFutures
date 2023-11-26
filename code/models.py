import tensorflow as tf
tf.random.set_seed(42)
# import tensorflow.keras.backend as K
# import tensorflow.keras.layers as layers
import tensorflow.python.keras.layers as layers
import tensorflow.python.keras.backend as K
# from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def create_ae_mlp(num_columns, num_labels, hidden_units, dropout_rates, ls=1e-2, lr=1e-3):
    inp = tf.keras.layers.Input(shape=(num_columns,))
    x0 = tf.keras.layers.BatchNormalization()(inp)

    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('swish')(encoder)

    decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
    decoder = tf.keras.layers.Dense(num_columns, name='decoder')(decoder)

    x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation('swish')(x_ae)
    x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)

    out_ae = tf.keras.layers.Dense(num_labels, activation='sigmoid', name='ae_action')(x_ae)

    x = tf.keras.layers.Concatenate()([x0, encoder])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)

    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)

    out = tf.keras.layers.Dense(num_labels, activation='softmax', name='action')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=[decoder, out_ae, out])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss={'decoder': tf.keras.losses.MeanSquaredError(),
                        'ae_action': tf.keras.losses.SparseCategoricalCrossentropy(),
                        'action': tf.keras.losses.SparseCategoricalCrossentropy(),
                        },
                  metrics={'decoder': tf.keras.metrics.MeanAbsoluteError(name='MAE'),
                           'ae_action': tf.keras.metrics.Accuracy(name='ACC'),
                           'action': tf.keras.metrics.Accuracy(name='ACC'),
                           },
                  )

    return model


def create_model(n_in, n_out, layers, dropout_rate, optimizer, metrics):
    inp = tf.keras.layers.Input(shape=(n_in,))

    x = inp
    for i, hidden_units in enumerate(layers):
        x = tf.keras.layers.BatchNormalization()(x)
        if i > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        else:
            x = tf.keras.layers.Dropout(.01)(x)
        x = tf.keras.layers.Dense(hidden_units)(x)
        x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Dense(n_out)(x)
    out = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=metrics,
                  #                   run_eagerly=True
                  )

    return model