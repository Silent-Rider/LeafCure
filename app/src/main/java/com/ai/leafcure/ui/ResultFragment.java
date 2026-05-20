package com.ai.leafcure.ui;

import android.graphics.Bitmap;
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

import com.ai.leafcure.R;
import com.ai.leafcure.databinding.FragmentResultBinding;
import com.bumptech.glide.Glide;
import dagger.hilt.android.AndroidEntryPoint;

@AndroidEntryPoint
public class ResultFragment extends Fragment {

    private FragmentResultBinding binding;
    private PredictionResult result;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentResultBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        result = new PredictionResult();
        result.setDiseaseName("Early Blight");
        result.setConfidence(0.95f);
        result.setSeverity(0.35f);

        setupUI();
        setupClickListeners();
    }

    private void setupUI() {
        if (result == null) return;

        binding.diseaseName.setText(result.getDiseaseName());
        binding.confidence.setText(String.format("Вероятность: %.1f%%", result.getConfidence() * 100));

        int severityPercent = (int) (result.getSeverity() * 100);
        binding.progressSeverity.setProgress(severityPercent);
        binding.severityValue.setText(severityPercent + "%");

        String imageUriString = getArguments() != null ? getArguments().getString("image_uri") : null;
        if (imageUriString != null) {
            Glide.with(this)
                    .load(Uri.parse(imageUriString))
                    .into(binding.imageOriginal);
        }

        if (result.getMaskBitmap() != null) {
            Bitmap redMask = colorizeMask(result.getMaskBitmap());
            binding.spotMask.setImageBitmap(redMask);
        } else {
            binding.isRequiredOrders.setEnabled(false);
            binding.isRequiredOrders.setText("Маска недоступна");
        }
    }

    private void setupClickListeners() {
        binding.isRequiredOrders.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                binding.spotMask.setVisibility(View.VISIBLE);
                binding.spotMask.animate().alpha(0.7f).setDuration(300).start();
            } else {
                binding.spotMask.animate().alpha(0.0f).setDuration(300).withEndAction(() -> {
                    binding.spotMask.setVisibility(View.GONE);
                }).start();
            }
        });
        binding.treatment.setOnClickListener(v -> showTreatmentBottomSheet());

        binding.newDiagnosis.setOnClickListener(v -> {
            Navigation.findNavController(v).popBackStack(R.id.home_page, false);
        });
    }

    private void showTreatmentBottomSheet() {
        TreatmentBottomSheetFragment bottomSheet = new TreatmentBottomSheetFragment();
        Bundle args = new Bundle();
        args.putString("disease_name", result.getDiseaseName());
        bottomSheet.setArguments(args);

        bottomSheet.show(getChildFragmentManager(), "TreatmentBottomSheet");
    }

    private Bitmap colorizeMask(Bitmap mask) {
        if (mask == null) return null;

        Bitmap coloredMask = mask.copy(mask.getConfig(), true);
        int width = coloredMask.getWidth();
        int height = coloredMask.getHeight();

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int pixel = coloredMask.getPixel(x, y);
                if (Color.red(pixel) > 100 || Color.green(pixel) > 100 || Color.blue(pixel) > 100) {
                    coloredMask.setPixel(x, y, Color.argb(180, 255, 0, 0));
                } else {
                    coloredMask.setPixel(x, y, Color.TRANSPARENT);
                }
            }
        }
        return coloredMask;
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
