package com.ai.leafcure.data.dao;

import androidx.room.Dao;
import androidx.room.Query;

@Dao
public interface TreatmentDao {

    @Query("SELECT content FROM Treatment WHERE diseaseId = :diseaseId")
    String getContentByDiseaseId(Integer diseaseId);
}
