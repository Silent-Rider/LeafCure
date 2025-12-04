from keras.src.applications.mobilenet_v3 import preprocess_input
from keras.src.legacy.preprocessing.image import ImageDataGenerator

def get_train_val_generators(target_size:tuple, validation_split:float, batch_size:int):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=validation_split
    )

    train_generator = datagen.flow_from_directory(
        'dataset',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        'dataset',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, val_generator