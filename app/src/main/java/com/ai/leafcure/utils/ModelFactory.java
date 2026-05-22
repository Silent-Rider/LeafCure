package com.ai.leafcure.utils;

import android.content.Context;

import com.ai.leafcure.ml.BinaryClassifier;
import com.ai.leafcure.ml.CategoricalClassifier;
import com.ai.leafcure.ml.LeafSegmenter;
import com.ai.leafcure.ml.SpotSegmenter;

import javax.inject.Inject;
import javax.inject.Singleton;

import dagger.hilt.android.qualifiers.ApplicationContext;

@Singleton
public class ModelFactory {

    private final Context context;
    private final ImageUtils imageUtils;

    @Inject
    public ModelFactory(@ApplicationContext Context context, ImageUtils imageUtils) {
        this.context = context;
        this.imageUtils = imageUtils;
    }

    public BinaryClassifier createBinaryClassifier(String plantName) {
        return new BinaryClassifier(context, imageUtils, plantName);
    }

    public CategoricalClassifier createCategoricalClassifier(String plantName) {
        return new CategoricalClassifier(context, imageUtils, plantName);
    }

    public LeafSegmenter createLeafSegmenter() {
        return new LeafSegmenter(context, imageUtils);
    }

    public SpotSegmenter createSpotSegmenter() {
        return new SpotSegmenter(context, imageUtils);
    }
}
