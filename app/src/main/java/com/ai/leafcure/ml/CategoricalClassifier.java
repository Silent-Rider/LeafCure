package com.ai.leafcure.ml;

import android.content.Context;
import android.graphics.Bitmap;

import com.ai.leafcure.utils.ImageUtils;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

public class CategoricalClassifier extends BaseModel {

    private final int numClasses;
    private final Map<Integer, String> classIndexToName;

    public CategoricalClassifier(Context context, String plantName) throws Exception {
        super(context, "classify/" + plantName + "_categorical.tflite");
        this.numClasses = interpreter.getOutputTensor(0).shape()[1];
        this.classIndexToName = loadClassIndices(context, plantName + "_categorical.txt");
    }

    private Map<Integer, String> loadClassIndices(Context context, String fileName) throws Exception {
        Map<Integer, String> map = new HashMap<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open("meta/" + fileName)));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(": ", 2);
            if (parts.length == 2) {
                int index = Integer.parseInt(parts[0].trim());
                String name = parts[1].trim();
                map.put(index, name);
            }
        }
        reader.close();
        return map;
    }

    public float[] predictDiseaseType(Bitmap bitmap) {
        if (bitmap.getWidth() != inputSize || bitmap.getHeight() != inputSize) {
            throw new IllegalArgumentException("Bitmap size mismatch.");
        }

        ByteBuffer inputBuffer = ImageUtils.bitmapToFloatBuffer(bitmap);

        float[][] outputArray = new float[1][numClasses];

        interpreter.run(inputBuffer, outputArray);

        return outputArray[0];
    }

    public int getNumClasses() {
        return numClasses;
    }

    public String getClassName(int index) {
        return classIndexToName.getOrDefault(index, "Unknown");
    }
}
