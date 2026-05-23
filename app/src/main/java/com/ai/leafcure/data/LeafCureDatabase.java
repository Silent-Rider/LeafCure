package com.ai.leafcure.data;

import androidx.room.Database;
import androidx.room.RoomDatabase;

import com.ai.leafcure.data.dao.DiseaseDao;
import com.ai.leafcure.data.dao.PlantDao;
import com.ai.leafcure.data.dao.TreatmentDao;
import com.ai.leafcure.data.entity.Disease;
import com.ai.leafcure.data.entity.Plant;
import com.ai.leafcure.data.entity.Treatment;

@Database(entities = {
        Plant.class,
        Disease.class,
        Treatment.class
}, version = 1)
public abstract class LeafCureDatabase extends RoomDatabase {

    public abstract PlantDao plantDao();
    public abstract DiseaseDao diseaseDao();
    public abstract TreatmentDao treatmentDao();
}
