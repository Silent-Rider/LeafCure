from keras import Model, regularizers
from keras.src.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.src.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large
from keras.src.metrics.f_score_metrics import F1Score as F1Score
from tensorflow.keras.models import load_model

from process.data_gen import get_train_val_generators
from keras.src.applications.mobilenet_v3 import preprocess_input

MODELS_FOLDER = 'models'

def create_mobile_net_v3_large(img_size:tuple):
    mobile_net = MobileNetV3Large(
        input_shape=(*img_size, 3),
        weights='imagenet',
        include_top=False
    )
    mobile_net.trainable = False
    return mobile_net, preprocess_input


def compile_model(base_model, num_classes=38):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    f1_score = F1Score(average='macro')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', f1_score]
    )
    return model


def fit_and_save_model(model:Model, train_gen, val_gen, epochs:int, version:int, prefix:str=""):
    if prefix: prefix += '_'
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.evaluate(val_gen)
    folder = MODELS_FOLDER
    model.save(f"{folder}/{prefix}model_v{version}.keras")
    model.export(f"{folder}/{prefix}model_v{version}")


def load_fitted_model(model_path:str):
    folder = MODELS_FOLDER
    model:Model = load_model(filepath=f"{folder}/{model_path}")
    return model


def main():
    image_size = (224, 224)
    base_model, preprocess_input_function = create_mobile_net_v3_large(image_size)
    model = compile_model(base_model)
    train_gen, val_gen = get_train_val_generators('dataset',
                                                  image_size,
                                                  preprocess_input_function,
                                                  validation_split=0.15,
                                                  batch_size=32,
                                                  seed=43)
    fit_and_save_model(model, train_gen, val_gen, epochs=10, version=1)
