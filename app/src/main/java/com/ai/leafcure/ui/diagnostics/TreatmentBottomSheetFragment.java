package com.ai.leafcure.ui.diagnostics;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.android.material.bottomsheet.BottomSheetDialogFragment;
import com.ai.leafcure.databinding.BottomSheetTreatmentBinding;

import dagger.hilt.android.AndroidEntryPoint;

@AndroidEntryPoint
public class TreatmentBottomSheetFragment extends BottomSheetDialogFragment {

    private BottomSheetTreatmentBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = BottomSheetTreatmentBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        Bundle args = getArguments();
        if (args != null) {
            String diseaseName = args.getString("disease_name");
            binding.diseaseName.setText(diseaseName);
            String treatment = args.getString("treatment");
            binding.description.setText(treatment);
        }
        binding.close.setOnClickListener(v -> dismiss());
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
