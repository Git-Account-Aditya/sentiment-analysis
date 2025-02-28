import numpy as np
from tensorflow.keras.layers import SimpleRNN, Dense, Input, Embedding, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Load IMDB data
vocab_size = 20000  # Limit vocabulary size
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Set maximum sequence length
max_len = 250
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

# Model architecture
input_layer = Input(shape=(max_len,))
embedding = Embedding(input_dim=vocab_size + 1, output_dim=64)(input_layer)
lstm = LSTM(128, dropout=0.3, return_sequences=False)(embedding)
dense = Dense(64, activation='relu')(lstm)
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['precision', 'recall', 'accuracy'])

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])

# Save the model
model.save('sentiment_analysis.keras')