import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf
# from tf.keras import layers, models, optimizers, losses
import numpy as np
import datetime
from lib import metrics, utils  # Assuming this is a custom module provided

data = utils.load_dataset("data/METR-LA", 64, 64)
scaler = data['scaler']
train_loader = data['train_loader']
test_loader = data['test_loader']
val_loader = data['val_loader']

OUT_STEPS = 3
num_features = 20


# multi_lstm_model = tf.keras.Sequential([
#     # Shape [batch, time, features] => [batch, lstm_units].
#     # Adding more `lstm_units` just overfits more quickly.
#     tf.keras.layers.LSTM(32, return_sequences=False),
#     # Shape => [batch, out_steps*features].
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros()),
#     # Shape => [batch, out_steps, features].
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ])
#
# history = compile_and_fit(multi_lstm_model, multi_window)
#
# IPython.display.clear_output()
#
# multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val, return_dict=True)
# multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0, return_dict=True)
# multi_window.plot(multi_lstm_model)


# def _prepare_data(x, y):
#     x, y = _get_x_y(x, y)
#     x, y = _get_x_y_in_correct_dims(x, y)
#     return tf.convert_to_tensor(x), tf.convert_to_tensor(y)
#
# def _get_x_y(x, y):
#     """
#     :param x: shape (batch_size, seq_len, num_sensor, input_dim)
#     :param y: shape (batch_size, horizon, num_sensor, input_dim)
#     :returns x shape (seq_len, batch_size, num_sensor, input_dim)
#              y shape (horizon, batch_size, num_sensor, input_dim)
#     """
#     x = tf.convert_to_tensor(x, dtype=tf.float32)
#     y = tf.convert_to_tensor(y, dtype=tf.float32)
#     x = tf.transpose(x, perm=[1, 0, 2, 3])
#     # y = tf.transpose(y, perm=[1, 0, 2, 3])
#     return x, y
#
# def _get_x_y_in_correct_dims(x, y):
#     """
#     :param x: shape (seq_len, batch_size, num_sensor, input_dim)
#     :param y: shape (horizon, batch_size, num_sensor, input_dim)
#     :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
#              y: shape (horizon, batch_size, num_sensor * output_dim)
#     """
#     # batch_size = tf.shape(x)[1]
#     # x = tf.reshape(x[..., :1], [3, batch_size, 20 * 1])
#     # y = tf.reshape(y[..., :1], [3, batch_size, 20 * 1])
#     return x, y
#
# def _compute_loss(y_true, y_predicted):
#     y_true = scaler.inverse_transform(y_true.numpy())
#     y_predicted = scaler.inverse_transform(y_predicted.numpy())
#     return metrics.masked_mae_loss(y_predicted, y_true)
#
# def evaluate(dataset='val', batches_seen=0):
#     """
#     Computes mean L1Loss
#     :return: mean L1Loss
#     """
#     val_iterator = data['{}_loader'.format(dataset)].get_iterator()
#     losses = []
#
#     y_truths = []
#     y_preds = []
#     for _, (x, y) in enumerate(val_iterator):
#         x, y = _prepare_data(x, y)
#
#         output = model(x)
#         loss = _compute_loss(y, output)
#         losses.append(loss)
#
#         y_truths.append(y.numpy())
#         y_preds.append(output.numpy())
#
#     mean_loss = np.mean(losses)
#
#     y_preds = np.concatenate(y_preds, axis=1)
#     y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension
#
#     for t in range(y_preds.shape[0]):
#         y_truth = scaler.inverse_transform(y_truths[t])
#         y_pred = scaler.inverse_transform(y_preds[t])
#
#         mae = metrics.masked_mae_np(y_pred, y_truth, null_val=0)
#         mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
#         rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
#         if dataset != 'val':
#             print(
#                 "Horizon {:02d}, MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
#                     t + 1, mae, mape, rmse
#                 )
#             )
#
#     return mean_loss
#
# class CustomTrafficLSTM(tf.keras.Model):
#     def __init__(self, units, out_steps):
#         super(CustomTrafficLSTM, self).__init__()
#         self.out_steps = out_steps
#         self.lstm_cell = layers.LSTMCell(units)
#         self.lstm_rnn = layers.RNN(self.lstm_cell, return_state=True)
#         # self.lstm = layers.LSTM(hidden_size, num_layers, return_sequences=True, dropout=0.5)
#         self.dense = layers.Dense(20)
#
#     def call(self, inputs, training=None):
#         # Use a TensorArray to capture dynamically unrolled outputs.
#         predictions = []
#         # Initialize the LSTM state.
#         prediction, state = self.warmup(inputs)
#
#         # Insert the first prediction.
#         predictions.append(prediction)
#
#         # Run the rest of the prediction steps.
#         for n in range(1, self.out_steps):
#             # Use the last prediction as input.
#             x = prediction
#             # Execute one lstm step.
#             x, state = self.lstm_cell(x, states=state,
#                                       training=training)
#             # Convert the lstm output to a prediction.
#             prediction = self.dense(x)
#             # Add the prediction to the output.
#             predictions.append(prediction)
#
#         # predictions.shape => (time, batch, features)
#         predictions = tf.stack(predictions)
#         # predictions.shape => (batch, time, features)
#         predictions = tf.transpose(predictions, [1, 0, 2])
#         return predictions
#
#
#     def warmup(self, inputs):
#         # inputs.shape => (batch, time, features)
#         # x.shape => (batch, lstm_units)
#         x, * state = self.lstm_rnn(inputs)
#
#         # predictions.shape => (batch, features)
#         prediction = self.dense(x)
#         return prediction, state
#
#
# # Initialize the model
# lstm_units = 20
#
# model = CustomTrafficLSTM(units=32, out_steps=3)
# optimizer = optimizers.Adam(learning_rate=0.001)
# mae_criterion = losses.MeanAbsoluteError()
#
#
# prediction, state = model.warmup(multi_window.example[0])
#
# prediction.shape
#
#
#
#
#
#
# # Training
# best_val_loss = float('inf')
# patience = 1
# trigger_times = 0
#
# train_mae = []
# val_mae = []
#
# for epoch in range(500):
#     current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
#     #print(f'{current_time} - Epoch {epoch + 1} started')
#     train_loss = 0
#     train_mae_accum = 0
#
#     for _, (inputs, labels) in enumerate(train_loader.get_iterator()):
#         x, y = _prepare_data(inputs, labels)
#         with tf.GradientTape() as tape:
#             outputs = model(x)
#             loss = _compute_loss(y, outputs)
#         grads = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         train_loss += loss
#         train_mae_accum += mae_criterion(y, outputs).numpy()
#
#     val_loss = evaluate("val")
#
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         trigger_times = 0
#     else:
#         trigger_times += 1
#         if trigger_times >= patience:
#             print("Early stopping!")
#             break
#
#     #print(f'Epoch {epoch + 1}, Val Loss: {val_loss}')
#
# # Test evaluation
# test_loss = 0
# test_mae_accum = 0
# result = evaluate('test')
# print(f'MAE {result}')
