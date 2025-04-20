# src/model_training.py


from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
from data_prep import preprocess

def train_model(features):
    input_dim = features.shape[1]
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    embedding = layers.Dense(8, name='embedding')(x)
    x = layers.Dense(16, activation='relu')(embedding)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(input_dim)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    model.fit(features, features, epochs=50, batch_size=8, verbose=1)

    encoder = Model(inputs=inp, outputs=embedding)
    return model, encoder.predict(features)

if __name__ == "__main__":
    df, features = preprocess("C:/Users/Karthikraj/Downloads/coding/ml/rs/data/amz_ca.csv")
    model, embeddings = train_model(features)
    model.save("models/autoencoder_model.h5")
    np.save("models/product_embeddings.npy", embeddings)
    df.to_csv("data/products_with_embeddings.csv", index=False)
