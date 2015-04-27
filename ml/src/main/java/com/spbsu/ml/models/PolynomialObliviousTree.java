package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Func;
import com.spbsu.ml.methods.trees.GreedyObliviousLinearTree;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 20:50
 * To change this template use File | Settings | File Templates.
 */
public class PolynomialObliviousTree extends Func.Stub {
  protected final BFGrid.BinaryFeature[] features;
  protected final double[][] values;

  public PolynomialObliviousTree(final BFGrid.BinaryFeature[] features, final double[][] values) {
    this.features = features;
    this.values = values;
  }

  @Override
  public double value(final Vec point) {
    final int region = ObliviousTree.bin(features, point);
    double[] factors = GreedyObliviousLinearTree.getSignificantFactors(point, features);
    double sum = 0;
    for (int i = 0; i < factors.length; i++) {
      for (int j = 0; j <= i; j++) {
        sum += values[region][i * (i + 1) / 2 + j] * factors[i] * factors[j];
      }
    }
    return sum;
  }

  @Override
  public int dim() {
    return features[0].row().grid().rows();
  }

  String indexToTexLetteral(final int i) {
    if (i == 0) {
      return "1";
    } else {
      return "x_{" + features[i - 1].findex + "}";
    }
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    for (int mask = 0; mask < 1 << features.length; mask++) {

      for (int i = 0; i < features.length; i++) {
        builder.
            append("$x_{").
            append(features[i].findex).
            append("}").
            append(((mask >> i) & 1) == 0 ? " < " : " > ").
            append(features[i].condition).
            append("$  ");
      }

      builder.append("\n$");
      for (int i = 0; i <= features.length; i++) {
        for (int j = 0; j <= i; j++) {
          builder.
              append(values[mask][i * (i + 1) / 2 + j]).
              append(" * ").
              append(indexToTexLetteral(i)).
              append(" * ").
              append(indexToTexLetteral(j)).
              append(" + ");
        }
      }

      builder.append("$\n");
    }
    return builder.toString();
  }
}
