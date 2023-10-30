from tensorflow import keras
import data_fitting
class nn_model(keras.Model):
    def __init__(self):
        super(nn_model, self).__init__()
        self.input_layer = keras.layers.Input(shape=(data_fitting.x_train.shape[1]))
        self.hidden1 = keras.layers.Dense(units=200, activation='relu', kernel_initializer="normal")
        self.dropout1 = keras.layers.Dropout(rate=0.15)
        self.hidden2 = keras.layers.Dense(units=100, activation='relu', kernel_initializer="normal")
        self.dropout2 = keras.layers.Dropout(rate=0.15)
        self.hidden3 = keras.layers.Dense(units=10, activation='relu', kernel_initializer="normal")
        self.output_layer = keras.layers.Dense(units=1)
    
    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.dropout1(x)
        x = self.hidden2(x)
        x = self.dropout2(x)
        x = self.hidden3(x)
        return self.output_layer(x)
    
    def compile_model(self):
        self.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["mae"])