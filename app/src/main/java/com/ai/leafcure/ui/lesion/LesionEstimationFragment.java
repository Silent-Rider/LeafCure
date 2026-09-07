package com.ai.leafcure.ui.lesion;

import android.os.Bundle;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.navigation.Navigation;

import com.ai.leafcure.R;
import com.ai.leafcure.ui.diagnostics.DiagnosticsFragment;

import dagger.hilt.android.AndroidEntryPoint;

@AndroidEntryPoint
public class LesionEstimationFragment extends DiagnosticsFragment {

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        binding.choose.setText(R.string.make_or_load_damaged_plant_photo);
        binding.cardPlantSelection.setVisibility(View.GONE);
        binding.start.setOnClickListener(v -> Navigation.findNavController(v)
                .navigate(LesionEstimationFragmentDirections.actionLesionEstimationToProcess(imageUri.toString(), null)));
        binding.layoutActions.setVisibility(View.VISIBLE);
    }
}
