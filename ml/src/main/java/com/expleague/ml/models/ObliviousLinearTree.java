package com.expleague.ml.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.BFGrid;
import com.expleague.ml.BinModelWithGrid;
import com.expleague.ml.BinOptimizedModel;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * User: solar
 * Date: 31.01.2020
 */
public class ObliviousLinearTree extends BinOptimizedModel.Stub implements BinModelWithGrid {
  private final BFGrid.Feature[] features;
  private final Vec[] weights;
  private final double[] basedOn;
  private final int[] projection;

  public ObliviousLinearTree(final List<BFGrid.Feature> features, final Vec[] weights) {
    this(features, weights, new double[weights.length]);
  }

  public ObliviousLinearTree(final List<BFGrid.Feature> features, final Vec[] weights, final double[] basedOn) {
    if (features.size() == 0)
      throw new RuntimeException("Creating oblivious tree of zero depth");
    this.basedOn = basedOn;
    this.features = features.toArray(new BFGrid.Feature[features.size()]);
    this.weights = weights;
    this.projection = features.stream()/*.filter(bf -> bf.row().size() > 2)*/.mapToInt(BFGrid.Feature::findex).sorted().distinct().toArray();
  }

  @Override
  public int dim() {
    return features[0].row().grid().size();
  }

  @Override
  public double value(final Vec x) {
    final int index = bin(x);
    return weights[index].get(0) + VecTools.multiply(weights[index].sub(1, projection.length), (Vec)x.sub(projection));
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(weights.length);
    builder.append("->(");
    for (int i = 0; i < features.length; i++) {
      builder.append(i > 0 ? ", " : "")
          .append(features[i]);
    }
    builder.append(")");
    builder.append("+[");
    for (int i = 0; i < weights.length; i++) {
      builder.append("{").append(weights[i]).append("}").append("@").append(basedOn[i]).append(", ");
    }
    builder.delete(builder.length() - 2, builder.length());
    builder.append("]");
    return builder.toString();
  }

  public int bin(final Vec x) {
    int index = 0;
    for (BFGrid.Feature feature : features) {
      index <<= 1;
      if (feature.value(x))
        index++;
    }
    return index;
  }

  public List<BFGrid.Feature> features() {
    return new ArrayList<>(Arrays.asList(features));
  }

  public Vec[] values() {
    return weights;
  }

  public double[] based() {
    return basedOn;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof ObliviousLinearTree)) return false;

    final ObliviousLinearTree that = (ObliviousLinearTree) o;

    if (!Arrays.equals(features, that.features)) return false;
    return Arrays.equals(weights, that.weights);
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(features);
    result = 31 * result + Arrays.hashCode(weights);
    result = 31 * result + Arrays.hashCode(basedOn);
    return result;
  }

  public BFGrid grid() {
    return features[0].row().grid();
  }

  @Override
  public double value(final BinarizedDataSet bds, final int pindex) {
    int index = 0;
    for (final BFGrid.Feature feature : features) {
      index <<= 1;
      if (feature.value(bds.bins(feature.findex())[pindex]))
        index++;
    }
    final Vec x = ((VecDataSet) bds.original()).data().row(pindex);
    return weights[index].get(0) + VecTools.multiply(weights[index].sub(1, projection.length), (Vec)x.sub(projection));
  }
}
