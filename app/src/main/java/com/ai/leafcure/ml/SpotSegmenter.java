package com.ai.leafcure.ml;

import android.content.Context;
import android.graphics.Bitmap;

import com.ai.leafcure.utils.ImageUtils;

import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

public class SpotSegmenter extends BaseModel {

    public SpotSegmenter(Context context) throws Exception {
        super(context, "segment/spot_seg.tflite");
    }

    public float[][] segmentSpots(Bitmap bitmap) {
        if (bitmap.getWidth() != inputSize || bitmap.getHeight() != inputSize) {
            throw new IllegalArgumentException("Bitmap size mismatch.");
        }

        ByteBuffer inputBuffer = ImageUtils.bitmapToFloatBuffer(bitmap);

        float[][][][] outputArray = new float[1][inputSize][inputSize][1];

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, outputArray);

        interpreter.runForMultipleInputsOutputs(new Object[]{inputBuffer}, outputs);

        float[][] mask = new float[inputSize][inputSize];
        for (int y = 0; y < inputSize; y++) {
            for (int x = 0; x < inputSize; x++) {
                mask[y][x] = outputArray[0][y][x][0];
            }
        }
        return mask;
    }
}
