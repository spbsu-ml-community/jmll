package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Model;

import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 20:50
 * To change this template use File | Settings | File Templates.
 */
public class ContinousObliviousTree extends Model {
    private final BFGrid.BinaryFeature[] features;
    private final double[][] values;
    //private final double[] basedOn;
    //private final double score;

    public ContinousObliviousTree(final List<BFGrid.BinaryFeature> features, double[][] values)/*, double[] basedOn, double bestScore)*/ {
        //For every leaf you must make pass 1 value for 0 degree coefficient, n - for 1 degree coefficient, n^2 for 2 degree, and so on
        assert values.length == 1 << features.size();
        for (int i = 0; i < values.length; i++)
            assert values[i].length == features.size() * features.size() + 2 * features.size() + 1;
        //this.basedOn = basedOn;
        this.features = features.toArray(new BFGrid.BinaryFeature[features.size()]);
        this.values = values;
        //this.score = bestScore;
    }

    @Override
    public double value(Vec _x) {
        int index = bin(_x);
        double sum = 0;
        double x[] = new double[features.length + 1];
        for(int i = 0; i < features.length; i++)
            x[i] = _x.get(i);
        x[features.length] = 1;
        for(int i = 0; i < features.length;i++)
            for(int j = 0;j < features.length;j++)
                sum += x[i] * x[j];
        return sum;
    }

    @Override
    public String toString() {
        /*StringBuilder builder = new StringBuilder();
        builder.append(values.length).append("/").append(basedOn);
        builder.append(" ->(");
        for (int i = 0; i < features.length; i++) {
            builder.append(i > 0 ? ", " : "")
                    .append(features[i]);
        }
        builder.append(")");
        return builder.toString();*/
        return "ContinousObliviousTree toString method didn't implemented yet";
    }

    public int bin(Vec x) {
        int index = 0;
        for (int i = 0; i < features.length; i++) {
            index <<= 1;
            if (features[i].value(x))
                index++;
        }
        return index;
    }
}
