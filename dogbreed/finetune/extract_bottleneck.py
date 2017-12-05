import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator


def extract_bottleneck(batch_size=16):
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

    model = VGG16(
        include_top=False,
        weights='imagenet', input_shape
        =(150, 150, 3))

    bottlenect_features_train = model.predict_generator(
        train_generator,
        steps=len(train_generator.filenames),
        verbose=1)

    bottlenect_features_valid = model.predict_generator(
        validation_generator,
        steps=len(validation_generator.filenames),
        verbose=1)

    return bottlenect_features_train, bottlenect_features_valid


if __name__ == "__main__":
    train, valid = extract_bottleneck()
    np.save("bottlenect_train.npy", train)
    np.save("bottlenect_valid.npy", valid)
