package com.ai.leafcure.ui.common;

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
import com.ai.leafcure.ui.BasicActivityViewModel;
import com.ai.leafcure.utils.ImageUtils;
import com.ai.leafcure.utils.MathUtils;
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
    @Inject
    MathUtils mathUtils;

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
        if (selectedPlant != null) {
            binding.processText.setText(R.string.diagnostics_in_process);
            runDiagnostics();
        } else {
            binding.processText.setText(R.string.lesion_estimation_in_process);
            runLesionEstimation();
        }
    }

    private void runDiagnostics() {
        boolean hasLeafMaskFunction = activityViewModel.hasLeafMaskFunction();
        executor.execute(() -> {
            try (LeafSegmenter leafSegmenter = modelFactory.createLeafSegmenter();
                 BinaryClassifier binaryClassifier = modelFactory.createBinaryClassifier(selectedPlant);
                 CategoricalClassifier categoricalClassifier = modelFactory.createCategoricalClassifier(selectedPlant);
                 SpotSegmenter spotSegmenter = modelFactory.createSpotSegmenter()) {

                Bitmap originalBitmap = imageUtils.loadBitmapFromUri(originalImageUri);
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true);
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
                    int predictedClassIndex = mathUtils.getMaxIndex(classProbabilities);
                    confidence = classProbabilities[predictedClassIndex];
                    String diseaseName = categoricalClassifier.getClassName(predictedClassIndex);

                    Bitmap maskedForSpots = imageUtils.applyMaskToBitmap(resizedBitmap, leafMask);
                    float[][] spotMask = spotSegmenter.segmentSpots(maskedForSpots);

                    severity = mathUtils.calculateSeverity(leafMask, spotMask);

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
            } catch (Exception e) {
                String errorText = "Ошибка диагностики: " + e.getMessage();
                requireActivity().runOnUiThread(() -> binding.processText.setText(errorText));
            }
        });
    }

    private void runLesionEstimation() {
        boolean hasLeafMaskFunction = activityViewModel.hasLeafMaskFunction();
        executor.execute(() -> {
            try (LeafSegmenter leafSegmenter = modelFactory.createLeafSegmenter();
                 SpotSegmenter spotSegmenter = modelFactory.createSpotSegmenter()) {

                Bitmap originalBitmap = imageUtils.loadBitmapFromUri(originalImageUri);
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true);
                float[][] leafMask = leafSegmenter.segmentLeaf(resizedBitmap);
                Uri leafMaskUri = null;

                if (hasLeafMaskFunction) {
                    Bitmap leafMaskBitmap = imageUtils.convertFloatMaskToBitmap(leafMask);
                    leafMaskUri = imageUtils.saveBitmapToTempFile(leafMaskBitmap, originalBitmap);
                }

                Bitmap maskedForSpots = imageUtils.applyMaskToBitmap(resizedBitmap, leafMask);
                float[][] spotMask = spotSegmenter.segmentSpots(maskedForSpots);

                float severity = mathUtils.calculateSeverity(leafMask, spotMask);

                Bitmap spotMaskBitmap = imageUtils.convertFloatMaskToBitmap(spotMask);
                Uri spotMaskUri = imageUtils.saveBitmapToTempFile(spotMaskBitmap, originalBitmap);

                String finalSpotMaskUri = spotMaskUri == null ? "" : spotMaskUri.toString();
                String finalLeafMaskUri = leafMaskUri == null ? null : leafMaskUri.toString();
                requireActivity().runOnUiThread(() -> {
                    NavOptions navOptions = new NavOptions.Builder()
                            .setPopUpTo(R.id.process, true)
                            .build();
                    Navigation.findNavController(binding.getRoot())
                            .navigate(ProcessFragmentDirections.actionProcessToResult(
                                    null,
                                    -1.0f,
                                    severity,
                                    originalImageUri.toString(),
                                    finalSpotMaskUri,
                                    finalLeafMaskUri
                            ), navOptions);
                });
            } catch (Exception e) {
                String errorText = "Ошибка диагностики: " + e.getMessage();
                requireActivity().runOnUiThread(() -> binding.processText.setText(errorText));
            }
        });
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        executor.shutdownNow();
        binding = null;
    }
}