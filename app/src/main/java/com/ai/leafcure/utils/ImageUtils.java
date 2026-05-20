package com.ai.leafcure.utils;

import android.graphics.Bitmap;
import android.graphics.Color;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Objects;

public class ImageUtils {
    public static ByteBuffer bitmapToFloatBuffer(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int channels = 3;

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * width * height * channels);
        byteBuffer.order(java.nio.ByteOrder.nativeOrder());
        FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < width * height; i++) {
            int pixel = pixels[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;

            floatBuffer.put((float) r);
            floatBuffer.put((float) g);
            floatBuffer.put((float) b);
        }

        return byteBuffer;
    }

    public static Bitmap colorizeMask(Bitmap mask) {
        if (mask == null) return null;

        Bitmap coloredMask = mask.copy(Objects.requireNonNull(mask.getConfig()), true);
        int width = coloredMask.getWidth();
        int height = coloredMask.getHeight();

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int pixel = coloredMask.getPixel(x, y);
                if (Color.red(pixel) > 50 || Color.green(pixel) > 50 || Color.blue(pixel) > 50) {
                    coloredMask.setPixel(x, y, Color.argb(180, 255, 0, 0));
                } else {
                    coloredMask.setPixel(x, y, Color.TRANSPARENT);
                }
            }
        }
        return coloredMask;
    }
}
