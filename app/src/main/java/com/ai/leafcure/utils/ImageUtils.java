package com.ai.leafcure.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
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

    public Bitmap loadResizedBitmapFromUri(Uri uri, int targetSize) {
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;

            InputStream inputStream = context.getContentResolver().openInputStream(uri);
            if (inputStream == null) return null;

            BitmapFactory.decodeStream(inputStream, null, options);
            inputStream.close();

            int srcWidth = options.outWidth;
            int srcHeight = options.outHeight;
            int sampleSize = 1;

            while ((srcWidth / sampleSize > targetSize) ||
                    (srcHeight / sampleSize > targetSize)) {
                sampleSize *= 2;
            }

            options.inJustDecodeBounds = false;
            options.inSampleSize = sampleSize;

            inputStream = context.getContentResolver().openInputStream(uri);
            if (inputStream == null) return null;

            Bitmap bitmap = BitmapFactory.decodeStream(inputStream, null, options);
            inputStream.close();

            if (bitmap != null && (bitmap.getWidth() != targetSize || bitmap.getHeight() != targetSize)) {
                Bitmap resized = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true);
                if (resized != bitmap) {
                    bitmap.recycle();
                }
                return resized;
            }

            return bitmap;

        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    public int[] getImageDimensions(Uri uri) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        try (InputStream is = context.getContentResolver().openInputStream(uri)) {
            BitmapFactory.decodeStream(is, null, options);
        } catch (IOException e) {
            throw new RuntimeException(e.getMessage());
        }
        return new int[]{options.outWidth, options.outHeight};
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

    public Uri saveBitmapToTempFile(Bitmap maskBitmap, int originalImageWidth, int originalImageHeight) throws Exception {
        Bitmap resizedMask = Bitmap.createScaledBitmap(
                maskBitmap,
                originalImageWidth,
                originalImageHeight,
                true
        );

        File cacheDir = context.getCacheDir();
        String tempFileName = String.format("mask_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("ddMMyyyyHHmmssSSS")));
        File tempFile = File.createTempFile(tempFileName, ".png", cacheDir);

        FileOutputStream fos = new FileOutputStream(tempFile);
        resizedMask.compress(Bitmap.CompressFormat.PNG, 100, fos);
        fos.flush();
        fos.close();

        return Uri.fromFile(tempFile);
    }

    public Bitmap colorizeMask(Bitmap mask, int color) {
        if (mask == null) return null;

        Bitmap coloredMask = mask.copy(Objects.requireNonNull(mask.getConfig()), true);
        int width = coloredMask.getWidth();
        int height = coloredMask.getHeight();

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int pixel = coloredMask.getPixel(x, y);
                if (Color.red(pixel) > 50 || Color.green(pixel) > 50 || Color.blue(pixel) > 50) {
                    coloredMask.setPixel(x, y, color != Color.BLACK ? color : Color.TRANSPARENT);
                } else {
                    coloredMask.setPixel(x, y, color != Color.BLACK ? Color.TRANSPARENT : color);
                }
            }
        }
        return coloredMask;
    }
}
