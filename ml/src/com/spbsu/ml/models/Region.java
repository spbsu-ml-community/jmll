package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Model;

import java.util.List;

/**
 * User: solar
 * Date: 29.11.12
 * Time: 5:35
 */
public class Region extends Model {
    private final int[] features;
    private final double[] conditions;
    private final boolean[] mask;
    private final double value;
    private final int basedOn;
    private final double score;

    public Region(final List<BinaryCond> conditions, double value, int basedOn, double bestScore) {
        this.basedOn = basedOn;
        this.features = new int[conditions.size()];
        this.conditions = new double[conditions.size()];
        this.mask = new boolean[conditions.size()];
        this.value = value;
        for (int i = 0; i < conditions.size(); i++) {
            this.features[i] = conditions.get(i).bf.findex;
            this.conditions[i] = conditions.get(i).bf.condition;
            this.mask[i] = conditions.get(i).mask;
        }
        this.score = bestScore;
    }

    @Override
    public double value(Vec x) {
        for (int i = 0; i < features.length; i++) {
            if ((x.get(features[i]) > conditions[i]) != mask[i])
                return 0.;
        }
        return value;
    }

    public boolean contains(Vec x) {
        for (int i = 0; i < features.length; i++) {
            if ((x.get(features[i]) > conditions[i]) != mask[i])
                return false;
        }

        return true;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(value).append("/").append(basedOn);
        builder.append(" ->");
        for (int i = 0; i < features.length; i++) {
            builder.append(" ")
                    .append(features[i])
                    .append(mask[i] ? ">" : "<=")
                    .append(conditions[i]);
        }
        return builder.toString();
    }

    public static class BinaryCond {
        public BFGrid.BinaryFeature bf;
        public boolean mask;

        public boolean yes(byte[] folds) {
            return bf.value(folds) == mask;
        }

        @Override
        public String toString() {
            StringBuilder builder = new StringBuilder();
            builder.append(" ")
                    .append(bf.findex)
                    .append(mask ? ">=" : "<")
                    .append(bf.condition);

            return builder.toString();
        }
    }
}
