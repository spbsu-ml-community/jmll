package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.commons.math.Func;

import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 20:50
 * To change this template use File | Settings | File Templates.
 */
public class ContinousObliviousTree extends Func.Stub {
    protected final BFGrid.BinaryFeature[] features;
    protected final double[][] values;
    //private final double[] basedOn;
    //private final double score;

    public ContinousObliviousTree(final List<BFGrid.BinaryFeature> features, final double[][] values)/*, double[] basedOn, double bestScore)*/ {
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
    public int dim() {
        return features[0].row().grid().size();
    }

    @Override
    public double value(final Vec _x) {
        final int index = bin(_x);
        double sum = 0;
        final double[] x = new double[features.length + 1];
        for (int i = 0; i < features.length; i++)
            x[i + 1] = _x.get(features[i].findex);
        x[0] = 1;
        for (int i = 0; i <= features.length; i++)
            for (int j = 0; j <= i; j++)
                sum += values[index][i * (i + 1) / 2 + j] * x[i] * x[j];
        return sum;
    }

    String indexToTexLetteral(final int i) {
        if (i == 0)
            return "1";
        else
            return "x_{" + features[i - 1].findex + "}";
    }

    @Override
    public String toString() {
        final StringBuilder builder = new StringBuilder();
        for (int mask = 0; mask < 1 << features.length; mask++) {

            for (int i = 0; i < features.length; i++)
                builder.append("$x_{" + (features[i].findex) + "}" + (((mask >> i) & 1) == 0 ? " < " : " > ") + features[i].condition + "$  ");

            builder.append("\n$");
            for (int i = 0; i <= features.length; i++)
                for (int j = 0; j <= i; j++) {
                    builder.append(values[mask][i * (i + 1) / 2 + j] + " * " + indexToTexLetteral(i) + " * " + indexToTexLetteral(j) + " + ");
                }

            builder.append("$\n");
        }
        return builder.toString();
    }

    public int bin(final Vec x) {
        int index = 0;
        for (int i = 0; i < features.length; i++) {
            index <<= 1;
            if (features[i].value(x))
                index++;
        }
        return index;
    }
}
