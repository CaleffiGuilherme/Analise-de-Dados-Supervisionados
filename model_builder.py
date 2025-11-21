import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_mlp_model(input_dim, activation_func='relu', learning_rate=0.001):
    model = Sequential()
    
    model.add(Dense(64, input_dim=input_dim, activation=activation_func))
    
    model.add(Dense(32, activation=activation_func))
    
    model.add(Dense(16, activation=activation_func))
    
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model