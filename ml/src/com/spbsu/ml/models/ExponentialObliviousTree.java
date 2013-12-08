package com.spbsu.ml.models;


import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;

import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 30.11.13
 * Time: 18:03
 * Idea please stop making my code yellow
 */
public class ExponentialObliviousTree extends ContinousObliviousTree {
    public ExponentialObliviousTree(final List<BFGrid.BinaryFeature> features, double[][] values) {
        super(features, values);
    }

    double sqr(double x) {
        return x * x;
    }

    double calcDistanseToRegion(int index, Vec point) {
        double ans = 0;
        for (int i = 0; i < features.length; i++) {
            if (features[i].value(point) != ((index >> i) == 1)) {
                ans += sqr(point.get(features[i].findex) - features[i].condition);//L2
            }
        }
        return 5 * ans;
    }

    @Override
    public double value(Vec _x) {
        double sum = 0;

        double x[] = new double[features.length + 1];
        for (int i = 0; i < features.length; i++)
            x[i + 1] = _x.get(features[i].findex);
        x[0] = 1;
        double sumWeights = 0;
        for (int index = 0; index < 1 << features.length; index++) {
            double weight = Math.exp(-calcDistanseToRegion(index, _x));
            sumWeights += weight;
            for (int i = 0; i <= features.length; i++)
                for (int j = 0; j <= i; j++)
                    sum += values[index][i * (i + 1) / 2 + j] * x[i] * x[j] * weight;
        }
        return sum;// / sumWeights;
    }
}
