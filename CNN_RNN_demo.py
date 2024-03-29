from data_preparation import load_and_process_data
from grad_cam import make_gradcam_heatmap, display_gradcam

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Load and process the data
data_path = 'E:/PHD Projects/Project 1/mitdb'
data, labels = load_and_process_data(data_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential()

# CNN layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(300, 1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

# LSTM layers
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dropout(0.5))

# Fully connected layers
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='softmax'))  # Assuming 5 classes for your problem

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, verbose=1, mode='auto')

# Train the model with a specified number of epochs
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# heatmap = make_gradcam_heatmap(your_test_image_array, your_model, 'last_conv_layer_name')
# display_gradcam(your_test_image_array, heatmap)

# Plotting the loss function over time
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Function Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate and print the classification report
report = classification_report(y_test, y_pred_classes, target_names=['N', 'V', 'S', 'F', 'Other'])
print(report)

model.save('E:/PHD Projects/Project 1/CNN_RNN_model.h5')



