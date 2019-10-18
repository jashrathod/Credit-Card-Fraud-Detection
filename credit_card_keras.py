import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pylab import rcParams
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from numpy.random import seed
  
seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
SEED = 123  # used to help randomly select the data points
DATA_SPLIT_PCT = 0.2
rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal", "Fraud"]

df = pd.read_csv("creditcard.csv")

# print(df.shape[0])

sign = lambda x: (1, -1)[x < 0]

df_train, df_test = train_test_split(df, test_size=DATA_SPLIT_PCT, random_state=SEED)
df_train, df_valid = train_test_split(df_train, test_size=DATA_SPLIT_PCT, random_state=SEED)

# print(df_train.shape[0], df_valid.shape[0], df_test.shape[0])

df_train_0 = df_train.loc[df['Class'] == 0]
df_train_1 = df_train.loc[df['Class'] == 1]
df_train_0_x = df_train_0.drop(['Class'], axis=1)
df_train_1_x = df_train_1.drop(['Class'], axis=1)
# print(df_train_0_x.shape[0], df_train_0_x.shape[0])

df_valid_0 = df_valid.loc[df['Class'] == 0]
df_valid_1 = df_valid.loc[df['Class'] == 1]
df_valid_0_x = df_valid_0.drop(['Class'], axis=1)
df_valid_1_x = df_valid_1.drop(['Class'], axis=1)
# print(df_valid_0_x.shape[0], df_valid_0_x.shape[0])

df_test_0 = df_test.loc[df['Class'] == 0]
df_test_1 = df_test.loc[df['Class'] == 1]
df_test_0_x = df_test_0.drop(['Class'], axis=1)
df_test_1_x = df_test_1.drop(['Class'], axis=1)
# print(df_test_0_x.shape[0], df_test_0_x.shape[0])

scaler = StandardScaler().fit(df_train_0_x)
df_train_0_x_rescaled = scaler.transform(df_train_0_x)
df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)
df_valid_x_rescaled = scaler.transform(df_valid.drop(['Class'], axis=1))
df_test_0_x_rescaled = scaler.transform(df_test_0_x)
df_test_x_rescaled = scaler.transform(df_test.drop(['Class'], axis=1))

nb_epoch = 100
batch_size = 128
input_dim = df_train_0_x_rescaled.shape[1]  # num of predictor variables,
encoding_dim = 32
hidden_dim = int(encoding_dim / 2)
learning_rate = 10e-8
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(learning_rate))(input_layer)
encoder = Dense(hidden_dim, activation='relu')(encoder)
decoder = Dense(hidden_dim, activation='relu')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = load_model('autoencoder_classifier.h5')

# autoencoder = Model(inputs=input_layer, outputs=decoder)
#
# autoencoder.compile(metrics=['accuracy'],
#                     loss='mean_squared_error',
#                     optimizer='adam')
# cp = ModelCheckpoint(filepath="autoencoder_classifier.h5",
#                      save_best_only=True,
#                      verbose=0)
# tb = TensorBoard(log_dir='./logs',
#                  histogram_freq=0,
#                  write_graph=True,
#                  write_images=True)
#
#
# history = autoencoder.fit(df_train_0_x_rescaled, df_train_0_x_rescaled,
#                           epochs=nb_epoch,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled),
#                           verbose=1,
#                           callbacks=[cp, tb]).history

valid_x_predictions = autoencoder.predict(df_valid_x_rescaled)
mse = np.mean(np.power(df_valid_x_rescaled - valid_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': df_valid['Class']})
precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

f1_array = []
max_f1_array = 0
precision_recall_pos = 0
for i in range(len(recall_rt)):
    num = 2 * recall_rt[i] * precision_rt[i]
    den = recall_rt[i] + precision_rt[i]
    temp_f1_array = float (num/den)
    f1_array.append(temp_f1_array)
    if temp_f1_array > max_f1_array:
        max_f1_array = temp_f1_array
        precision_recall_pos = i
print(max_f1_array, precision_recall_pos)
print(precision_rt[precision_recall_pos])
print(recall_rt[precision_recall_pos])

test_x_predictions = autoencoder.predict(df_test_x_rescaled)
mse = np.mean(np.power(df_test_x_rescaled - test_x_predictions, 2), axis=1)
print("mse: ", mse)
error_df_test = pd.DataFrame({'Reconstruction_error': mse, 'True_class': df_test['Class']})
error_df_test = error_df_test.reset_index()
threshold_fixed = 0.22
groups = error_df_test.groupby('True_class')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label="Fraud" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

pred_y = [1 if e > threshold_fixed else 0 for e in error_df_test.Reconstruction_error.values]
conf_matrix = confusion_matrix(error_df_test.True_class, pred_y)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# opt_th=0
# max_f1=0
# conf_matrix=np.zeros((2, 2))
# temp_conf_matrix=np.zeros((2, 2))
# j = np.arange(0, 5, 0.1)
# for i in j:
#     pred_y = [1 if e > i else 0 for e in error_df_test.Reconstruction_error.values]
#     conf_matrix = confusion_matrix(error_df_test.True_class, pred_y)
#     temp_recall = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])
#     temp_precision = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
#     temp_f1 = 2*temp_recall*temp_precision/(temp_precision + temp_recall)
#     if temp_f1 > max_f1:
#         opt_th = i
#         max_f1 = temp_f1
#         temp_conf_matrix = conf_matrix
#
# conf_matrix = temp_conf_matrix
# threshold_fixed = opt_th
# print(threshold_fixed)
# plt.figure(figsize=(12, 12))
# sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
# plt.title("Confusion matrix")
# plt.ylabel('True class')
# plt.xlabel('Predicted class')
# plt.show()

false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate,)
plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0, 1], [0, 1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
