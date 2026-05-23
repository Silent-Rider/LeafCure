package com.ai.leafcure.ui;

import static com.ai.leafcure.utils.ImageUtils.MODEL_INPUT_SIZE;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavOptions;
import androidx.navigation.Navigation;

import com.ai.leafcure.R;
import com.ai.leafcure.databinding.FragmentProcessBinding;
import com.ai.leafcure.ml.*;
import com.ai.leafcure.utils.ImageUtils;
import com.ai.leafcure.utils.ModelFactory;

import dagger.hilt.android.AndroidEntryPoint;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.inject.Inject;

@AndroidEntryPoint
public class ProcessFragment extends Fragment {

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private FragmentProcessBinding binding;
    private BasicActivityViewModel activityViewModel;
    private String selectedPlant;
    private Uri originalImageUri;

    @Inject
    ModelFactory modelFactory;
    @Inject
    ImageUtils imageUtils;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        activityViewModel = new ViewModelProvider(requireActivity()).get(BasicActivityViewModel.class);
        binding = FragmentProcessBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        ProcessFragmentArgs args = ProcessFragmentArgs.fromBundle(requireArguments());
        selectedPlant = args.getSelectedPlant();
        originalImageUri = Uri.parse(args.getImageUri());
        runDiagnostics();
    }

    private void runDiagnostics() {
        boolean hasLeafMaskFunction = activityViewModel.hasLeafMaskFunction();
        executor.execute(() -> {
            try {
                Bitmap originalBitmap = imageUtils.loadBitmapFromUri(originalImageUri);
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true);

                LeafSegmenter leafSegmenter = modelFactory.createLeafSegmenter();
                BinaryClassifier binaryClassifier = modelFactory.createBinaryClassifier(selectedPlant);
                CategoricalClassifier categoricalClassifier = modelFactory.createCategoricalClassifier(selectedPlant);
                SpotSegmenter spotSegmenter = modelFactory.createSpotSegmenter();

                float[][] leafMask = leafSegmenter.segmentLeaf(resizedBitmap);


                float healthScore = binaryClassifier.predictHealth(resizedBitmap);
                boolean isDiseased = healthScore < 0.5f;

                String diagnosisResult;
                float confidence;
                float severity = 0.0f;
                Uri spotMaskUri = null;
                Uri leafMaskUri = null;

                if (hasLeafMaskFunction) {
                    Bitmap leafMaskBitmap = imageUtils.convertFloatMaskToBitmap(leafMask);
                    leafMaskUri = imageUtils.saveBitmapToTempFile(leafMaskBitmap, originalBitmap);
                }

                if (isDiseased) {
                    float[] classProbabilities = categoricalClassifier.predictDiseaseType(resizedBitmap);
                    int predictedClassIndex = getMaxIndex(classProbabilities);
                    confidence = classProbabilities[predictedClassIndex];
                    String diseaseName = categoricalClassifier.getClassName(predictedClassIndex);

                    Bitmap maskedForSpots = imageUtils.applyMaskToBitmap(resizedBitmap, leafMask);
                    float[][] spotMask = spotSegmenter.segmentSpots(maskedForSpots);

                    severity = calculateSeverity(leafMask, spotMask);

                    Bitmap spotMaskBitmap = imageUtils.convertFloatMaskToBitmap(spotMask);
                    spotMaskUri = imageUtils.saveBitmapToTempFile(spotMaskBitmap, originalBitmap);

                    diagnosisResult = diseaseName;
                } else {
                    diagnosisResult = "Healthy";
                    confidence = 1.0f - healthScore;
                }

                String finalSpotMaskUri = spotMaskUri == null ? "" : spotMaskUri.toString();
                String finalLeafMaskUri = leafMaskUri == null ? null : leafMaskUri.toString();
                float finalSeverity = severity;
                requireActivity().runOnUiThread(() -> {
                    NavOptions navOptions = new NavOptions.Builder()
                            .setPopUpTo(R.id.process, true)
                            .build();
                    Navigation.findNavController(binding.getRoot())
                            .navigate(ProcessFragmentDirections.actionProcessToResult(
                                    diagnosisResult,
                                    confidence,
                                    finalSeverity,
                                    originalImageUri.toString(),
                                    finalSpotMaskUri,
                                    finalLeafMaskUri
                                    ), navOptions);
                });

                leafSegmenter.close();
                binaryClassifier.close();
                categoricalClassifier.close();
                spotSegmenter.close();

            } catch (Exception e) {
                requireActivity().runOnUiThread(() -> {
                    binding.processText.setText("Ошибка диагностики: " + e.getMessage());
                });
            }
        });
    }

    private int getMaxIndex(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private float calculateSeverity(float[][] leafMask, float[][] spotMask) {
        float leafArea = 0;
        float spotArea = 0;

        for (int y = 0; y < leafMask.length; y++) {
            for (int x = 0; x < leafMask[0].length; x++) {
                leafArea += leafMask[y][x];
                spotArea += spotMask[y][x];
            }
        }

        if (leafArea == 0) return 0;
        return spotArea / leafArea;
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        executor.shutdownNow();
        binding = null;
    }
}