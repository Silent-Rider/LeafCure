from keras import Model, regularizers
from keras.src.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.src.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large
from keras.src.metrics.f_score_metrics import F1Score as F1Score

from process.data_gen import get_train_val_generators

img_size = (224, 224)

base_model = MobileNetV3Large(
    input_shape=(*img_size, 3),
    weights='imagenet',
    include_top=False
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
x = Dropout(0.5)(x)
predictions = Dense(38, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

f1_score = F1Score(average='macro')
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', f1_score]
)

train_gen, val_gen = get_train_val_generators(img_size, 0.15, 32)

model.fit(train_gen, validation_data=val_gen, epochs=10)
model.evaluate(val_gen)
model.save('models/plant_disease_model_v1.keras')
model.export('models/plant_disease_model_v1')


