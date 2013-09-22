package com.spbsu.ml.methods.trees;

import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.Aggregator;

/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:16
 */
public interface BFLeaf extends Aggregator {
    BFLeaf split(BFGrid.BinaryFeature feature);

    int score(double[] likelihoods);

    int size();

    double alpha();
}
