package com.ai.leafcure.ui.diagnostics;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.SwitchCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.ai.leafcure.R;
import com.ai.leafcure.databinding.FragmentResultBinding;
import com.ai.leafcure.ui.TreatmentBottomSheetFragment;
import com.ai.leafcure.utils.ImageUtils;
import com.bumptech.glide.Glide;
import dagger.hilt.android.AndroidEntryPoint;

import android.widget.CompoundButton;
import android.widget.ImageView;

import javax.inject.Inject;

@AndroidEntryPoint
public class ResultFragment extends Fragment {

    private FragmentResultBinding binding;
    private String diseaseName;
    private float confidence;
    private float severity;
    private String originalImageUriString;
    private String spotMaskUriString;
    private String leafMaskUriString;
    @Inject
    ImageUtils imageUtils;

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
        leafMaskUriString = args.getLeafMaskUriString();

        setupUI();
        setupClickListeners();
    }

    private void setupUI() {
        binding.diseaseName.setText(diseaseName);
        @SuppressLint("DefaultLocale")
        String confidence = String.format("Вероятность: %.1f%%", this.confidence * 100);
        binding.confidence.setText(confidence);

        int severityPercent = (int) (severity * 100);
        binding.progressSeverity.setProgress(severityPercent);
        String severity = severityPercent + "%";
        binding.severityValue.setText(severity);

        if (originalImageUriString != null && !originalImageUriString.isEmpty()) {
            Glide.with(this)
                    .load(Uri.parse(originalImageUriString))
                    .into(binding.imageOriginal);
        }
        if (leafMaskUriString != null) {
            binding.emptyView.setVisibility(View.GONE);
            binding.applyLeafMask.setVisibility(View.VISIBLE);
        } else {
            binding.emptyView.setVisibility(View.VISIBLE);
            binding.applyLeafMask.setVisibility(View.GONE);
        }
        showMask(spotMaskUriString, Color.argb(180, 255, 0, 0), binding.spotMask, binding.showLesions);
        showMask(leafMaskUriString, Color.BLACK, binding.leafMask, binding.applyLeafMask);
    }

    private void setupClickListeners() {
        binding.showLesions.setOnCheckedChangeListener(getOnCheckedChangeListener(binding.spotMask));
        binding.applyLeafMask.setOnCheckedChangeListener(getOnCheckedChangeListener(binding.leafMask));
        binding.treatment.setOnClickListener(v -> showTreatmentBottomSheet());
        binding.newDiagnostics.setOnClickListener(v -> {
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

    private void showMask(String maskUriString, int color, ImageView maskView, SwitchCompat maskSwitch) {
        if (maskUriString != null && !maskUriString.isEmpty()) {
            try {
                Bitmap maskBitmap = imageUtils.loadBitmapFromUri(Uri.parse(maskUriString));
                if (maskBitmap != null) {
                    Bitmap blackMask = imageUtils.colorizeMask(maskBitmap, color);
                    maskView.setImageBitmap(blackMask);
                    maskSwitch.setEnabled(true);
                } else {
                    maskSwitch.setEnabled(false);
                    maskSwitch.setText("Маска недоступна");
                }
            } catch (Exception e) {
                maskSwitch.setEnabled(false);
                maskSwitch.setText("Ошибка загрузки маски");
            }
        }
    }

    private CompoundButton.OnCheckedChangeListener getOnCheckedChangeListener(ImageView maskView) {
        return (buttonView, isChecked) -> {
            if (isChecked) {
                maskView.setVisibility(View.VISIBLE);
                maskView.animate().alpha(0.7f).setDuration(300).start();
            } else {
                maskView.animate().alpha(0.0f).setDuration(300).withEndAction(() -> {
                    maskView.setVisibility(View.GONE);
                }).start();
            }
        };
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}