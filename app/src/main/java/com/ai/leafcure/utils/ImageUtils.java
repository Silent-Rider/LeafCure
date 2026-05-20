package com.ai.leafcure.utils;

import android.graphics.Bitmap;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

import javax.inject.Singleton;

public class ImageUtils {
    public static ByteBuffer bitmapToFloatBuffer(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int channels = 3; // RGB

        // Создаем буфер для float значений: H * W * C
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * width * height * channels);
        byteBuffer.order(java.nio.ByteOrder.nativeOrder());
        FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < width * height; i++) {
            int pixel = pixels[i];
            // Извлекаем R, G, B. Android Bitmap хранит в ARGB_8888
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;

            // Просто приводим к float, без нормализации
            floatBuffer.put((float) r);
            floatBuffer.put((float) g);
            floatBuffer.put((float) b);
        }

        return byteBuffer;
    }
}
