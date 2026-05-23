package com.ai.leafcure.data.entity;

import static androidx.room.ForeignKey.CASCADE;

import androidx.room.Entity;
import androidx.room.ForeignKey;
import androidx.room.PrimaryKey;

@Entity(foreignKeys = {
        @ForeignKey(onDelete = CASCADE, entity = Plant.class,
                parentColumns = "id", childColumns = "plantId")
})
public class Disease {
    @PrimaryKey
    private Integer id;
    private Integer plantId;
    private String russianName;
    private String fullEnglishName;

    public Disease() {}

    public Disease(String russianName) {
        this.russianName = russianName;
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public Integer getPlantId() {
        return plantId;
    }

    public void setPlantId(Integer plantId) {
        this.plantId = plantId;
    }

    public String getRussianName() {
        return russianName;
    }

    public void setRussianName(String russianName) {
        this.russianName = russianName;
    }

    public String getFullEnglishName() {
        return fullEnglishName;
    }

    public void setFullEnglishName(String fullEnglishName) {
        this.fullEnglishName = fullEnglishName;
    }
}
