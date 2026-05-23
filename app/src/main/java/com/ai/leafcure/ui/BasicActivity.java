package com.ai.leafcure.ui;

import android.os.Bundle;
import android.text.Html;
import android.text.Spanned;
import android.view.MenuItem;

import androidx.activity.OnBackPressedCallback;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.view.GravityCompat;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.navigation.NavController;
import androidx.navigation.NavDestination;
import androidx.navigation.NavGraph;
import androidx.navigation.fragment.NavHostFragment;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.ai.leafcure.R;
import com.ai.leafcure.data.LeafCureDatabase;
import com.ai.leafcure.data.entity.Disease;
import com.ai.leafcure.data.entity.Plant;
import com.ai.leafcure.databinding.ActivityBasicBinding;
import com.google.android.material.navigation.NavigationView;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import javax.inject.Inject;

import dagger.hilt.android.AndroidEntryPoint;

@AndroidEntryPoint
public class BasicActivity extends AppCompatActivity {
    private NavController navController;
    private AppBarConfiguration appBarConfiguration;
    @Inject
    LeafCureDatabase leafCureDatabase;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ActivityBasicBinding binding = ActivityBasicBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        setSupportActionBar(binding.toolbar);

        DrawerLayout drawer = binding.drawerLayout;
        NavigationView navigationView = binding.navView;
        appBarConfiguration = new AppBarConfiguration.Builder(
                R.id.diagnostics,
                R.id.lesion_estimation,
                R.id.settings)
                .setOpenableLayout(drawer)
                .build();

        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager().findFragmentById(R.id.nav_host_fragment);
        navController = Objects.requireNonNull(navHostFragment).getNavController();
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);
        NavigationUI.setupWithNavController(navigationView, navController);

        NavGraph navGraph = navController.getNavInflater().inflate(R.navigation.nav_graph);
        navGraph.setStartDestination(R.id.diagnostics);
        navController.setGraph(navGraph);

        MenuItem about = navigationView.getMenu().findItem(R.id.about);
        List<Plant> plantList = leafCureDatabase.plantDao().getAll();
        Map<Integer, List<Disease>> diseaseMap = leafCureDatabase.diseaseDao().getAll().stream()
                        .collect(Collectors.groupingBy(Disease::getPlantId));
        Spanned aboutText = buildAboutText(plantList, diseaseMap, getString(R.string.about_text));

        about.setOnMenuItemClickListener(item -> {
            new AlertDialog.Builder(this)
                    .setTitle(getString(R.string.about_button))
                    .setMessage(aboutText)
                    .setPositiveButton(getString(R.string.ok), (dialog, which) -> dialog.dismiss())
                    .create()
                    .show();
            return true;
        });

        OnBackPressedCallback callback = new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                if (drawer.isDrawerOpen(GravityCompat.START)) {
                    drawer.closeDrawer(GravityCompat.START);
                    return;
                }
                NavDestination currentDestination = navController.getCurrentDestination();
                if (currentDestination != null && currentDestination.getId() == R.id.process) {
                    return;
                }
                if (!navController.popBackStack()) {
                    setEnabled(false);
                    getOnBackPressedDispatcher().onBackPressed();
                    setEnabled(true);
                }
            }
        };
        getOnBackPressedDispatcher().addCallback(this, callback);
    }

    @Override
    public boolean onSupportNavigateUp() {
        return NavigationUI.navigateUp(navController, appBarConfiguration) || super.onSupportNavigateUp();
    }

    private static Spanned buildAboutText(List<Plant> plantList,
                                          Map<Integer, List<Disease>> diseaseMap,
                                          String authorInfo) {
        StringBuilder sb = new StringBuilder();
        sb.append("<b><font color='#333333'>Поддерживаемые культуры и болезни:</font></b><br><br>");
        for (Plant plant : plantList) {
            sb.append("<font color='#2E7D32'><b>")
                    .append(plant.getRussianName())
                    .append("</b></font><br>");

            List<Disease> diseases = diseaseMap.getOrDefault(plant.getId(), null);

            if (diseases != null && !diseases.isEmpty()) {
                for (Disease disease : diseases) {
                    sb.append("&nbsp;&nbsp;&nbsp;&bull; ")
                            .append(disease.getRussianName())
                            .append("<br>");
                }
            } else {
                sb.append("&nbsp;&nbsp;&nbsp;<i>Нет данных о болезнях</i><br>");
            }
            sb.append("<br>");
        }
        sb.append("<hr><br>");
        sb.append("<font color='#666666'>")
                .append(authorInfo)
                .append("<br><br>")
                .append("© 2026 LeafCure. Все права защищены.")
                .append("</font>");
        return Html.fromHtml(sb.toString(), Html.FROM_HTML_MODE_LEGACY);
    }
}
