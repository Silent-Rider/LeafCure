package com.ai.leafcure.ui;

import androidx.lifecycle.ViewModel;

public class BasicActivityViewModel extends ViewModel {

    private boolean hasLeafMaskFunction;

    public boolean hasLeafMaskFunction() {
        return hasLeafMaskFunction;
    }

    public void setHasLeafMaskFunction(boolean hasLeafMaskFunction) {
        this.hasLeafMaskFunction = hasLeafMaskFunction;
    }
}
