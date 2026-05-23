package com.ai.leafcure.ui.diagnostics;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import com.google.android.material.bottomsheet.BottomSheetDialogFragment;
import com.ai.leafcure.databinding.BottomSheetTreatmentBinding;
import java.util.HashMap;
import java.util.Map;

public class TreatmentBottomSheetFragment extends BottomSheetDialogFragment {

    private BottomSheetTreatmentBinding binding;
    private static final Map<String, String> TREATMENT_ADVICE = new HashMap<>();
    static {
        TREATMENT_ADVICE.put("Early Blight", "1. Удалите пораженные листья.\n2. Обработайте медьсодержащими фунгицидами.\n3. Избегайте полива дождеванием.");
        TREATMENT_ADVICE.put("Late Blight", "1. Срочно удалите все больные растения.\n2. Используйте системные фунгициды.\n3. Проветривайте теплицу.");
        TREATMENT_ADVICE.put("Healthy", "Растение здорово! Продолжайте регулярный полив и внесение удобрений.");
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = BottomSheetTreatmentBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        String diseaseName = getArguments() != null ? getArguments().getString("disease_name") : "Неизвестно";
        binding.diseaseName.setText(diseaseName);

        String advice = TREATMENT_ADVICE.getOrDefault(diseaseName,
                "Конкретные рекомендации для этого заболевания отсутствуют. Обратитесь к агроному.");

        binding.description.setText(advice);
        binding.close.setOnClickListener(v -> dismiss());
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
