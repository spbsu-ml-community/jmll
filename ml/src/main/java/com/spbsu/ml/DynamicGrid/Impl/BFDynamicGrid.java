package com.spbsu.ml.DynamicGrid.Impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.DynamicGrid.Interface.BinaryFeature;
import com.spbsu.ml.DynamicGrid.Interface.DynamicGrid;
import com.spbsu.ml.DynamicGrid.Interface.DynamicRow;

public class BFDynamicGrid implements DynamicGrid {
    private DynamicRow[] rows;
    private final DynamicRow leastNonEmptyRow;
//    private ArrayList<BinaryFeature> features = new ArrayList<>();

    public BFDynamicGrid(DynamicRow[] rows) {
        this.rows = rows;
        for (DynamicRow row : rows) {
            row.setOwner(this);
        }

        DynamicRow least = null;
        for (int f = 0; f < rows.length; ++f)
            if (!rows[f].empty()) {
                least = rows[f];
                break;
            }
        leastNonEmptyRow = least;
    }

    public DynamicRow row(int feature) {
        return feature < rows.length ? rows[feature] : null;
    }

    @Override
    public void binarize(Vec x, int[] folds) {
        for (int i = 0; i < x.dim(); i++) {
            folds[i] = rows[i].bin(x.get(i));
        }

    }

    @Override
    public BinaryFeature bf(int fIndex, int binNo) {
        return rows[fIndex].bf(binNo);
    }

    @Override
    public DynamicRow nonEmptyRow() {
        return leastNonEmptyRow;
    }

    @Override
    public boolean addSplit(int feature) {
        return rows[feature].addSplit();
    }


    @Override
    public int[] hist() {
        int[] counts = new int[rows.length];
        for (int f = 0; f < rows.length; ++f) {
            counts[f] = rows[f].size();
        }
        return counts;
    }

    public int rows() {
        return rows.length;
    }


    @Override
    public boolean isActive(int fIndex, int binNo) {
        return bf(fIndex, binNo).isActive();
    }


    public DynamicRow[] allRows() {
        return rows;
    }


}