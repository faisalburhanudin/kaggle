import numpy as np
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


def create_model(batch_size=16):
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    validation_generator = datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    num_classes = len(train_generator.class_indices)

    train_data = np.load('bottleneck_features_train.npy')
    train_labels = to_categorical(train_generator.classes, num_classes=num_classes)

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = to_categorical(validation_generator.classes, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=50,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              verbose=1)

    model.save_weights('bottleneck_fc_model.h5')


if __name__ == "__main__":
    create_model()
