package com.spbsu.ml.DynamicGrid.Impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.DynamicGrid.Interface.BinaryFeature;
import com.spbsu.ml.DynamicGrid.Interface.DynamicRow;

import java.util.Comparator;

/**
 * Created by noxoomo on 23/07/14.
 */
public class BinaryFeatureImpl implements BinaryFeature {
    public final DynamicRow bfRow;
    private int binNo;
    public final int origFIndex;
    public final int borderIndex;
    private int useCount;

    public int useCount() {
        return useCount;
    }

    public void use() {
        ++useCount;
    }


    public final double condition;
    private boolean active = false;
    private double regScore = 0;

    public void setRegScore(double reg) {
        this.regScore = reg;
    }


    public BinaryFeatureImpl(MedianRow bfRow, int origFIndex, double condition, int index) {
        this.bfRow = bfRow;
        this.origFIndex = origFIndex;
        this.condition = condition;
        this.borderIndex = index;
    }


    public void setBinNo(int newBinNo) {
        this.binNo = newBinNo;
    }


    public void setActive(boolean status) {
        this.active = status;
    }

    @Override
    public double regularization() {
        if (active) return 0;
        return regScore;
    }

    public boolean value(Vec vec) {
        return vec.get(origFIndex) > condition;
    }

    public DynamicRow row() {
        return bfRow;
    }

    @Override
    public int binNo() {
        return binNo;
    }

    @Override
    public int fIndex() {
        return origFIndex;
    }


    @Override
    public boolean isActive() {
        return active;
    }


//    @Override
//    public boolean equals(Object o) {
//        if (this == o) return true;
//        if (!(o instanceof BinaryFeatureImpl)) return false;
//        BinaryFeatureImpl that = (BinaryFeatureImpl) o;
//        return fIndex() == that.fIndex() && bfRow.equals(that.bfRow);
//    }

//    @Override
//    public int hashCode() {
//        int result = bfRow.hashCode();
//        result = 31 * result;
//        return result;
//    }

    @Override
    public String toString() {
        return String.format("f[%d] > %g", origFIndex, condition);
    }


    static Comparator<BinaryFeatureImpl> conditionComparator = new Comparator<BinaryFeatureImpl>() {
        @Override
        public int compare(BinaryFeatureImpl a, BinaryFeatureImpl b) {
            return Double.compare(a.condition, b.condition);
        }
    };

    static Comparator<BinaryFeatureImpl> borderComparator = new Comparator<BinaryFeatureImpl>() {
        @Override
        public int compare(BinaryFeatureImpl a, BinaryFeatureImpl b) {
            return Integer.compare(a.borderIndex, b.borderIndex);
        }
    };


    public int gridHash;
}
