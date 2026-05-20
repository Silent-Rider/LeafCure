package com.ai.leafcure.ui;

import android.graphics.Bitmap;

public class PredictionResult {
    private String diseaseName;
    private float confidence;
    private float severity;
    private Bitmap maskBitmap;

    public PredictionResult() {
    }

    public PredictionResult(String diseaseName, float confidence, float severity, Bitmap maskBitmap) {
        this.diseaseName = diseaseName;
        this.confidence = confidence;
        this.severity = severity;
        this.maskBitmap = maskBitmap;
    }

    public String getDiseaseName() {
        return diseaseName;
    }

    public void setDiseaseName(String diseaseName) {
        this.diseaseName = diseaseName;
    }

    public float getConfidence() {
        return confidence;
    }

    public void setConfidence(float confidence) {
        this.confidence = confidence;
    }

    public float getSeverity() {
        return severity;
    }

    public void setSeverity(float severity) {
        this.severity = severity;
    }

    public Bitmap getMaskBitmap() {
        return maskBitmap;
    }

    public void setMaskBitmap(Bitmap maskBitmap) {
        this.maskBitmap = maskBitmap;
    }
}
