package com.ai.leafcure.data.dao;
import androidx.room.Dao;
import androidx.room.Query;

import com.ai.leafcure.data.entity.Plant;

import java.util.List;

@Dao
public interface PlantDao {

    @Query("SELECT * FROM Plant ORDER BY russianName")
    List<Plant> getAll();
}
