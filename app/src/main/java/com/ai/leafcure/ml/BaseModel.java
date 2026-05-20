package com.ai.leafcure.ml;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public abstract class BaseModel {
    protected Interpreter interpreter;
    protected int inputSize;

    public BaseModel(Context context, String modelPath) throws Exception {
        this.interpreter = loadModelFile(context, modelPath);
        this.inputSize = interpreter.getInputTensor(0).shape()[1];
    }

    private Interpreter loadModelFile(Context context, String modelPath) throws Exception {
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
        }
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
        }
    }
}
