package com.ai.leafcure.ui;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.FileProvider;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.ai.leafcure.R;
import com.ai.leafcure.databinding.FragmentHomeBinding;
import com.bumptech.glide.Glide;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import dagger.hilt.android.AndroidEntryPoint;

@AndroidEntryPoint
public class HomePageFragment extends Fragment {
    private static final Map<String, String> langPlantNameMap = new HashMap<String, String> () {{
        put("Яблоко", "apple");
        put("Кукуруза", "corn");
        put("Виноград", "grape");
        put("Картофель", "potato");
        put("Томат", "tomato");
    }};

    private FragmentHomeBinding binding;
    private List<String> plantList;

    private String selectedPlant = null;
    private Uri imageUri;

    private final ActivityResultLauncher<Intent> cameraLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            this::handleCameraResult
    );
    private final ActivityResultLauncher<String> galleryLauncher = registerForActivityResult(
            new ActivityResultContracts.GetContent(),
            this::handleGalleryResult
    );
    private final ActivityResultLauncher<String> permissionLauncher = registerForActivityResult(
            new ActivityResultContracts.RequestPermission(),
            this::handlePermissionResult
    );

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        plantList = Arrays.asList(getResources().getStringArray(R.array.plants));
    }

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentHomeBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        setupAutoCompleteTextView();
        setupClickListeners();

        binding.start.setOnClickListener(v -> {
            String selectedPlant = langPlantNameMap.get(this.selectedPlant);
            Navigation.findNavController(v).navigate(HomePageFragmentDirections.actionHomePageToProcess(selectedPlant, imageUri.toString()));
        });
    }

    private void setupAutoCompleteTextView() {
        AutoCompleteTextView autoComplete = binding.plantAutoComplete;
        ArrayAdapter<String> adapter = new ArrayAdapter<>(requireContext(), R.layout.item_dropdown, plantList);
        autoComplete.setAdapter(adapter);

        autoComplete.setOnItemClickListener((parent, view, position, id) -> {
            selectedPlant = parent.getItemAtPosition(position).toString();
            binding.layoutActions.setVisibility(View.VISIBLE);
            imageUri = null;
            binding.image.setVisibility(View.GONE);
            binding.start.setVisibility(View.GONE);
        });
    }

    private void setupClickListeners() {
        binding.camera.setOnClickListener(v -> {
            if (getActivity() != null) {
                permissionLauncher.launch(Manifest.permission.CAMERA);
            }
        });
        binding.gallery.setOnClickListener(v -> {
            galleryLauncher.launch("image/*");
        });
    }

    @SuppressLint("QueryPermissionsNeeded")
    private void openCamera() {
        try {
            File imageFile = createImageFile();

            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            Uri imageUri = FileProvider.getUriForFile(requireContext(),
                    requireContext().getPackageName() + ".provider",
                    imageFile);
            this.imageUri = imageUri;

            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
            cameraLauncher.launch(takePictureIntent);
        } catch (IOException ex) {
            Toast.makeText(getContext(), "Ошибка создания файла: " + ex.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private File createImageFile() throws IOException {
        String timeStamp = String.valueOf(System.currentTimeMillis());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = requireContext().getExternalFilesDir(null);
        return File.createTempFile(imageFileName,".jpg",storageDir);
    }

    private void processImage(Uri imageUri) {
        Glide.with(requireContext())
                .load(imageUri)
                .into(binding.image);
        binding.image.setVisibility(View.VISIBLE);
        binding.start.setVisibility(View.VISIBLE);
    }

    private void handleCameraResult(ActivityResult result) {
        if (result.getResultCode() == Activity.RESULT_OK) {
            Toast.makeText(getContext(), "Фото сделано: " + selectedPlant, Toast.LENGTH_SHORT).show();
            processImage(imageUri);
        } else {
            Toast.makeText(getContext(), "Съемка отменена", Toast.LENGTH_SHORT).show();
        }
    }

    private void handleGalleryResult(Uri uri) {
        if (uri != null) {
            Toast.makeText(getContext(), "Фото выбрано: " + selectedPlant, Toast.LENGTH_SHORT).show();
            imageUri = uri;
            processImage(uri);
        } else {
            Toast.makeText(getContext(), "Фото не выбрано", Toast.LENGTH_SHORT).show();
        }
    }

    private void handlePermissionResult(Boolean isGranted) {
        if (isGranted) {
            openCamera();
        } else {
            Toast.makeText(getContext(), "Нет разрешения на камеру", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
