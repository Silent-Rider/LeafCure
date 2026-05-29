package com.ai.leafcure.ui.settings;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.ai.leafcure.databinding.FragmentSettingsBinding;
import com.ai.leafcure.ui.BasicActivityViewModel;

import dagger.hilt.android.AndroidEntryPoint;

@AndroidEntryPoint
public class SettingsFragment extends Fragment {

    private FragmentSettingsBinding binding;
    private BasicActivityViewModel activityViewModel;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        activityViewModel = new ViewModelProvider(requireActivity()).get(BasicActivityViewModel.class);
        binding = FragmentSettingsBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }


    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        binding.showLesions.setOnCheckedChangeListener((buttonView, isChecked) ->
                activityViewModel.setHasLeafMaskFunction(true));
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
