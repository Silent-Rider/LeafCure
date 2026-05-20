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
import com.ai.leafcure.utils.ImageUtils;
import com.bumptech.glide.Glide;
import dagger.hilt.android.AndroidEntryPoint;
import java.io.InputStream;
import java.util.Objects;

import android.graphics.BitmapFactory;

@AndroidEntryPoint
public class ResultFragment extends Fragment {

    private FragmentResultBinding binding;

    private String diseaseName;
    private float confidence;
    private float severity;
    private String originalImageUriString;
    private String spotMaskUriString;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentResultBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        ResultFragmentArgs args = ResultFragmentArgs.fromBundle(requireArguments());
        diseaseName = args.getDiseaseName();
        confidence = args.getConfidence();
        severity = args.getSeverity();
        originalImageUriString = args.getOriginalImageUriString();
        spotMaskUriString = args.getSpotMaskUriString();

        setupUI();
        setupClickListeners();
    }

    private void setupUI() {
        binding.diseaseName.setText(diseaseName);
        binding.confidence.setText(String.format("Вероятность: %.1f%%", confidence * 100));

        int severityPercent = (int) (severity * 100);
        binding.progressSeverity.setProgress(severityPercent);
        binding.severityValue.setText(severityPercent + "%");

        if (originalImageUriString != null && !originalImageUriString.isEmpty()) {
            Glide.with(this)
                    .load(Uri.parse(originalImageUriString))
                    .into(binding.imageOriginal);
        }

        if (spotMaskUriString != null && !spotMaskUriString.isEmpty()) {
            try {
                Bitmap maskBitmap = loadBitmapFromUri(Uri.parse(spotMaskUriString));
                if (maskBitmap != null) {
                    Bitmap redMask = ImageUtils.colorizeMask(maskBitmap);
                    binding.spotMask.setImageBitmap(redMask);
                    binding.isRequiredOrders.setEnabled(true);
                } else {
                    binding.isRequiredOrders.setEnabled(false);
                    binding.isRequiredOrders.setText("Маска недоступна");
                }
            } catch (Exception e) {
                e.printStackTrace();
                binding.isRequiredOrders.setEnabled(false);
                binding.isRequiredOrders.setText("Ошибка загрузки маски");
            }
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
        args.putString("disease_name", diseaseName);
        bottomSheet.setArguments(args);

        bottomSheet.show(getChildFragmentManager(), "TreatmentBottomSheet");
    }

    private Bitmap loadBitmapFromUri(Uri uri) throws Exception {
        InputStream inputStream = requireContext().getContentResolver().openInputStream(uri);
        return BitmapFactory.decodeStream(inputStream);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}