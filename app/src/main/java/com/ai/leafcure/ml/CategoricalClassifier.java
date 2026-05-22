package com.ai.leafcure.ml;

import android.content.Context;
import android.graphics.Bitmap;

import com.ai.leafcure.utils.ImageUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

public class CategoricalClassifier extends BaseModel {

    private final int numClasses;
    private final Map<Integer, String> classIndexToName;

    public CategoricalClassifier(Context context, ImageUtils imageUtils, String plantName) {
        super(context, imageUtils, "classify/" + plantName + "_categorical.tflite");
        this.numClasses = interpreter.getOutputTensor(0).shape()[1];
        this.classIndexToName = loadClassIndices(plantName + "_categorical.txt");
    }

    private Map<Integer, String> loadClassIndices(String fileName) {
        Map<Integer, String> map = new HashMap<>();
        try (InputStreamReader inputStreamReader = new InputStreamReader(context.getAssets().open("meta/" + fileName));
             BufferedReader reader = new BufferedReader(inputStreamReader)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(": ", 2);
                if (parts.length == 2) {
                    int index = Integer.parseInt(parts[0].trim());
                    String name = parts[1].trim();
                    map.put(index, name);
                }
            }
            return map;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public float[] predictDiseaseType(Bitmap bitmap) {
        if (bitmap.getWidth() != inputSize || bitmap.getHeight() != inputSize) {
            throw new IllegalArgumentException("Bitmap size mismatch.");
        }
        ByteBuffer inputBuffer = imageUtils.bitmapToFloatBuffer(bitmap);
        float[][] outputArray = new float[1][numClasses];
        interpreter.run(inputBuffer, outputArray);
        return outputArray[0];
    }

    public String getClassName(int index) {
        return classIndexToName.getOrDefault(index, "Unknown");
    }
}
