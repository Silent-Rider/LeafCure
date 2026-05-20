package com.ai.leafcure.ui;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;
import com.ai.leafcure.databinding.FragmentProcessBinding;
import com.ai.leafcure.ml.*;
import dagger.hilt.android.AndroidEntryPoint;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@AndroidEntryPoint
public class ProcessFragment extends Fragment {

    private FragmentProcessBinding binding;
    private String selectedPlant;
    private Uri originalImageUri;

    private static final int MODEL_INPUT_SIZE = 256;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentProcessBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        ProcessFragmentArgs args = ProcessFragmentArgs.fromBundle(requireArguments());
        selectedPlant = args.getSelectedPlant();
        originalImageUri = Uri.parse(args.getImageUri());

        runDiagnosis();
    }

    private void runDiagnosis() {
        executor.execute(() -> {
            try {
                Bitmap originalBitmap = loadBitmapFromUri(originalImageUri);

                Bitmap resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true);

                LeafSegmenter leafSegmenter = new LeafSegmenter(requireContext());
                BinaryClassifier binaryClassifier = new BinaryClassifier(requireContext(), selectedPlant);
                CategoricalClassifier categoricalClassifier = new CategoricalClassifier(requireContext(), selectedPlant);
                SpotSegmenter spotSegmenter = new SpotSegmenter(requireContext());

                float[][] leafMask = leafSegmenter.segmentLeaf(resizedBitmap);

                float healthScore = binaryClassifier.predictHealth(resizedBitmap);
                boolean isDiseased = healthScore < 0.5f;

                String diagnosisResult;
                float confidence;
                float severity = 0.0f;
                Bitmap spotMaskBitmap;
                Uri spotMaskUri = null;

                if (isDiseased) {
                    float[] classProbabilities = categoricalClassifier.predictDiseaseType(resizedBitmap);
                    int predictedClassIndex = getMaxIndex(classProbabilities);
                    confidence = classProbabilities[predictedClassIndex];
                    String diseaseName = categoricalClassifier.getClassName(predictedClassIndex);

                    Bitmap maskedForSpots = applyMaskToBitmap(resizedBitmap, leafMask);
                    float[][] spotMask = spotSegmenter.segmentSpots(maskedForSpots);

                    severity = calculateSeverity(leafMask, spotMask);

                    spotMaskBitmap = convertFloatMaskToBitmap(spotMask);

                    spotMaskUri = saveBitmapToTempFile(spotMaskBitmap);

                    diagnosisResult = diseaseName;
                } else {
                    diagnosisResult = "Healthy";
                    confidence = 1.0f - healthScore;
                }

                Bundle bundle = new Bundle();
                bundle.putString("diagnosis", diagnosisResult);
                bundle.putFloat("confidence", confidence);
                bundle.putFloat("severity", severity);
                bundle.putString("original_image_uri", originalImageUri.toString());

                String finalSpotMaskUri;
                if (spotMaskUri != null) {
                    finalSpotMaskUri = spotMaskUri.toString();
                } else {
                    finalSpotMaskUri = "";
                }

                bundle.putString("plant", selectedPlant);
                float finalSeverity = severity;
                requireActivity().runOnUiThread(() -> {
                    Navigation.findNavController(binding.getRoot())
                            .navigate(ProcessFragmentDirections.actionProcessToResult(
                                    diagnosisResult,
                                    confidence,
                                    finalSeverity,
                                    originalImageUri.toString(),
                                    finalSpotMaskUri
                                    ));
                });

                leafSegmenter.close();
                binaryClassifier.close();
                categoricalClassifier.close();
                spotSegmenter.close();

            } catch (Exception e) {
                e.printStackTrace();
                requireActivity().runOnUiThread(() -> {
                    binding.processText.setText("Ошибка диагностики: " + e.getMessage());
                });
            }
        });
    }


    private Bitmap loadBitmapFromUri(Uri uri) throws Exception {
        InputStream inputStream = requireContext().getContentResolver().openInputStream(uri);
        return BitmapFactory.decodeStream(inputStream);
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

    private Bitmap applyMaskToBitmap(Bitmap src, float[][] mask) {
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

    private Bitmap convertFloatMaskToBitmap(float[][] mask) {
        Bitmap bitmap = Bitmap.createBitmap(ProcessFragment.MODEL_INPUT_SIZE, ProcessFragment.MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888);
        for (int y = 0; y < ProcessFragment.MODEL_INPUT_SIZE; y++) {
            for (int x = 0; x < ProcessFragment.MODEL_INPUT_SIZE; x++) {
                int grayValue = (int) (mask[y][x] * 255);
                bitmap.setPixel(x, y, Color.argb(255, grayValue, grayValue, grayValue));
            }
        }
        return bitmap;
    }

    private Uri saveBitmapToTempFile(Bitmap bitmap) throws Exception {
        File cacheDir = requireContext().getCacheDir();
        File tempFile = File.createTempFile("spot_mask_", ".png", cacheDir);

        FileOutputStream fos = new FileOutputStream(tempFile);
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
        fos.flush();
        fos.close();

        return Uri.fromFile(tempFile);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        executor.shutdownNow();
        binding = null;
    }
}