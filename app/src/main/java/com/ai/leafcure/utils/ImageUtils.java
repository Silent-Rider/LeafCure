package com.ai.leafcure.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Objects;

import javax.inject.Inject;
import javax.inject.Singleton;

import dagger.hilt.android.qualifiers.ApplicationContext;

@Singleton
public class ImageUtils {

    public static final int MODEL_INPUT_SIZE = 256;
    private final Context context;
    
    @Inject
    public ImageUtils(@ApplicationContext Context context) {
        this.context = context;
    }

    public Bitmap loadBitmapFromUri(Uri uri) throws Exception {
        InputStream inputStream = context.getContentResolver().openInputStream(uri);
        return BitmapFactory.decodeStream(inputStream);
    }
    
    public ByteBuffer bitmapToFloatBuffer(Bitmap bitmap) {
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

    public Bitmap colorizeMask(Bitmap mask) {
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

    public Bitmap applyMaskToBitmap(Bitmap src, float[][] mask) {
        Bitmap result = src.copy(Objects.requireNonNull(src.getConfig()), true);
        int w = src.getWidth();
        int h = src.getHeight();

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (mask[y][x] < 0.5f) {
                    result.setPixel(x, y, Color.BLACK);
                }
            }
        }
        return result;
    }

    public Bitmap convertFloatMaskToBitmap(float[][] mask) {
        Bitmap bitmap = Bitmap.createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888);
        for (int y = 0; y < MODEL_INPUT_SIZE; y++) {
            for (int x = 0; x < MODEL_INPUT_SIZE; x++) {
                int grayValue = (int) (mask[y][x] * 255);
                bitmap.setPixel(x, y, Color.argb(255, grayValue, grayValue, grayValue));
            }
        }
        return bitmap;
    }

    public Uri saveBitmapToTempFile(Bitmap spotMaskBitmap, Bitmap originalImage) throws Exception {
        Bitmap resizedMask = Bitmap.createScaledBitmap(
                spotMaskBitmap,
                originalImage.getWidth(),
                originalImage.getHeight(),
                true
        );

        File cacheDir = context.getCacheDir();
        File tempFile = File.createTempFile("spot_mask_", ".png", cacheDir);

        FileOutputStream fos = new FileOutputStream(tempFile);
        resizedMask.compress(Bitmap.CompressFormat.PNG, 100, fos);
        fos.flush();
        fos.close();

        return Uri.fromFile(tempFile);
    }
}
