package com.ai.leafcure.ml;

import android.content.Context;
import android.graphics.Bitmap;

import com.ai.leafcure.utils.ImageUtils;

import java.nio.ByteBuffer;

public class BinaryClassifier extends BaseModel {

    public BinaryClassifier(Context context, String plantName) throws Exception {
        super(context, "classify/" + plantName + "_binary.tflite");
    }

    public float predictHealth(Bitmap bitmap) {
        if (bitmap.getWidth() != inputSize || bitmap.getHeight() != inputSize) {
            throw new IllegalArgumentException("Bitmap size mismatch.");
        }

        ByteBuffer inputBuffer = ImageUtils.bitmapToFloatBuffer(bitmap);

        // Выходная форма для бинарной классификации: [1, 1]
        float[][] outputArray = new float[1][1];

        interpreter.run(inputBuffer, outputArray);

        return outputArray[0][0];
    }
}
