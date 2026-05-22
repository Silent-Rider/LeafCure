package com.ai.leafcure.ml;

import android.content.Context;
import android.graphics.Bitmap;

import com.ai.leafcure.utils.ImageUtils;

import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

public class LeafSegmenter extends BaseModel {

    public LeafSegmenter(Context context, ImageUtils imageUtils) {
        super(context, imageUtils, "segment/leaf_seg.tflite");
    }

    public float[][] segmentLeaf(Bitmap bitmap) {
        if (bitmap.getWidth() != inputSize || bitmap.getHeight() != inputSize) {
            throw new IllegalArgumentException("Bitmap size mismatch. Expected " + inputSize + "x" + inputSize);
        }
        ByteBuffer inputBuffer = imageUtils.bitmapToFloatBuffer(bitmap);
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
