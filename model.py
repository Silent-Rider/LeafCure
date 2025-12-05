from keras import Model, regularizers
from keras.src.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.src.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large
from keras.src.metrics.f_score_metrics import F1Score as F1Score

from process.data_gen import get_train_val_generators

def created_mobile_net_v3_large(img_size:tuple):
    mobile_net_model = MobileNetV3Large(
        input_shape=(*img_size, 3),
        weights='imagenet',
        include_top=False
    )
    mobile_net_model.trainable = False
    return mobile_net_model


def create_model(base_deep_model):
    x = base_deep_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(38, activation='softmax')(x)

    target_model = Model(inputs=base_deep_model.input, outputs=predictions)

    f1_score = F1Score(average='macro')
    target_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', f1_score]
    )
    return target_model


image_size = (224, 224)
base_model = created_mobile_net_v3_large(image_size)
model = create_model(base_model)
train_gen, val_gen = get_train_val_generators(image_size, 0.15, 32)

model.fit(train_gen, validation_data=val_gen, epochs=10)
model.evaluate(val_gen)
model.save('models/plant_disease_model_v1.keras')
model.export('models/plant_disease_model_v1')


