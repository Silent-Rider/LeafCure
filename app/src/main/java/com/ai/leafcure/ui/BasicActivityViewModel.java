package com.ai.leafcure.ui;

import androidx.lifecycle.ViewModel;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class BasicActivityViewModel extends ViewModel {

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private volatile boolean isProcessing = false;
    private boolean hasLeafMaskFunction;

    public ExecutorService getExecutor() {
        return executor;
    }

    public boolean isProcessing() { return isProcessing; }
    public void setProcessing(boolean processing) { this.isProcessing = processing; }

    public boolean hasLeafMaskFunction() {
        return hasLeafMaskFunction;
    }

    public void setHasLeafMaskFunction(boolean hasLeafMaskFunction) {
        this.hasLeafMaskFunction = hasLeafMaskFunction;
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        if (!executor.isShutdown()) {
            executor.shutdownNow();
        }
    }
}
