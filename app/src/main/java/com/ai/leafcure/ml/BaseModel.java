package com.ai.leafcure.ml;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import com.ai.leafcure.utils.ImageUtils;

import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public abstract class BaseModel {

    protected final Context context;
    protected final Interpreter interpreter;
    protected final ImageUtils imageUtils;
    protected final int inputSize;

    public BaseModel(Context context, ImageUtils imageUtils, String modelPath) {
        this.context = context;
        this.interpreter = loadModelFile(context, modelPath);
        this.imageUtils = imageUtils;
        this.inputSize = interpreter.getInputTensor(0).shape()[1];
    }

    private Interpreter loadModelFile(Context context, String modelPath)  {
        try (AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {

            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();

            MappedByteBuffer mappedByteBuffer = fileChannel.map(
                    FileChannel.MapMode.READ_ONLY,
                    startOffset,
                    declaredLength
            );

            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);

            return new Interpreter(mappedByteBuffer, options);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
        }
    }
}
