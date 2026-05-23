package com.ai.leafcure.utils;

import javax.inject.Inject;
import javax.inject.Singleton;

@Singleton
public class MathUtils {

    @Inject
    public MathUtils() {}

    public int getMaxIndex(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public float calculateSeverity(float[][] leafMask, float[][] spotMask) {
        float leafArea = 0;
        float spotArea = 0;

        for (int y = 0; y < leafMask.length; y++) {
            for (int x = 0; x < leafMask[0].length; x++) {
                leafArea += leafMask[y][x];
                spotArea += spotMask[y][x];
            }
        }

        if (leafArea == 0) return 0;
        return spotArea / leafArea;
    }
}
