package com.ai.leafcure.data.entity;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity
public class Plant {
    @PrimaryKey
    private Integer id;
    private String russianName;
    private String englishName;

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getRussianName() {
        return russianName;
    }

    public void setRussianName(String russianName) {
        this.russianName = russianName;
    }

    public String getEnglishName() {
        return englishName;
    }

    public void setEnglishName(String englishName) {
        this.englishName = englishName;
    }
}
