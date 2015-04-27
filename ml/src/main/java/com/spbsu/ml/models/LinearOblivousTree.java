package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Func;
import com.spbsu.ml.methods.trees.GreedyObliviousLinearTree;

import java.util.List;

/**
 * Created by towelenee on 4/16/15.
 */
public class LinearOblivousTree extends Func.Stub {
  private final BFGrid.BinaryFeature[] features;
  private final double[][] values;

  public LinearOblivousTree(final List<BFGrid.BinaryFeature> features, final double[][] values) {
    assert values.length == 1 << features.size();
    for (int i = 0;
         i < values.length;
         i++)
      assert values[i].length == features.size() + 1;
    this.features = features.toArray(new BFGrid.BinaryFeature[features.size()]);
    this.values = values;
  }

  @Override
  public int dim() {
    return features[0].row().grid().size();
  }

  @Override
  public double value(Vec x) {
    int bin = ObliviousTree.bin(features, x);
    double[] factors = GreedyObliviousLinearTree.getSignificantFactors(x, features);
    double sum = 0;
    for (int i = 0; i <= features.length; ++i)
      sum += factors[i] * values[bin][i];
    return sum;
  }
}
