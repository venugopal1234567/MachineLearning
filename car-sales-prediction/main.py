import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense


car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')
# print(car_df.head(5))
#print(car_df.tail(5))

X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
Y = car_df['Car Purchase Amount']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
#print(scaler.data_max_)

#print(scaler.data_min_)
Y = Y.values.reshape(-1, 1)
Y_scaled = scaler.fit_transform(Y)

#print(X_scaled.shape)
#Tain the Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size = 0.25)
#print(X_train.shape)

#Build the model with TensorFlow

model = Sequential()
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))
print(model.summary())

#Train
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=25, verbose=1, validation_split=0.2)

# epochs_hist.history.keys()
# plt.plot(epochs_hist.history['loss'])
# plt.plot(epochs_hist.history['val_loss'])
# plt.title('Model Loss Progress during training')
# plt.ylabel('Training and Validation loss')
# plt.xlabel('Epoch number')
# plt.legend(['Trining Loss', 'Validation Loss'])
#sns.pairplot(X)

#Gender, Age, Annual Salary, Credit Card Debit, Net Worth
X_test = np.array([[1,50, 50000, 10000,60000 ]])
Y_predict = model.predict(X_test)

print('Expected Purchase Amount', Y_predict)
#plt.show()