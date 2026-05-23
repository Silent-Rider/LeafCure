package com.ai.leafcure.data.dao;
import androidx.room.Dao;
import androidx.room.Query;

import com.ai.leafcure.data.entity.Disease;

import java.util.List;

@Dao
public interface DiseaseDao {

    @Query("SELECT * FROM Disease")
    List<Disease> getAll();

    @Query("SELECT * FROM Disease WHERE fullEnglishName = :fullEnglishName")
    Disease getByFullEnglishName(String fullEnglishName);
}
