package com.expleague.ml.models;

import com.expleague.commons.math.DiscontinuousTrans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.BFGrid;
import com.expleague.ml.BinModelWithGrid;
import com.expleague.ml.BinOptimizedModel;
import com.expleague.ml.data.impl.BinarizedDataSet;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * User: solar
 * Date: 29.11.12
 * Time: 5:35
 */
public class ObliviousTree extends BinOptimizedModel.Stub implements BinModelWithGrid {
  private final BFGrid.BinaryFeature[] features;
  private final double[] values;
  private final double[] basedOn;
  private final BFGrid grid;

  public ObliviousTree(final List<BFGrid.BinaryFeature> features, final double[] values) {
    this(features, values, new double[values.length]);
  }

  public ObliviousTree(final List<BFGrid.BinaryFeature> features, final double[] values, final double[] basedOn) {
    if (features.size() == 0)
      throw new RuntimeException("Creating oblivious tree of zero depth");
    grid = features.get(0).row().grid();
    this.basedOn = basedOn;
    this.features = features.toArray(new BFGrid.BinaryFeature[features.size()]);
    this.values = values;
  }

  @Override
  public int dim() {
    return grid.rows();
  }

  @Override
  public double value(final Vec x) {
    final int index = bin(x);
    return values[index];
  }

  @Override
  public DiscontinuousTrans subgradient() {
    return new ObliviousGrad();
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(values.length);
    builder.append("->(");
    for (int i = 0; i < features.length; i++) {
      builder.append(i > 0 ? ", " : "")
          .append(features[i]).append("@").append(basedOn[i]);
    }
    builder.append(")");
    builder.append("+[");
    for (final double feature : values) {
      builder.append(feature).append(", ");
    }
    builder.delete(builder.length() - 2, builder.length());
    builder.append("]");
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

  public List<BFGrid.BinaryFeature> features() {
    final List<BFGrid.BinaryFeature> ret = new ArrayList<BFGrid.BinaryFeature>();
    ret.addAll(Arrays.asList(features));
    return ret;
  }

  public double[] values() {
    return values;
  }

  public double[] based() {
    return basedOn;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof ObliviousTree)) return false;

    final ObliviousTree that = (ObliviousTree) o;

    if (!Arrays.equals(features, that.features)) return false;
    if (!Arrays.equals(values, that.values)) return false;

    return true;
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(features);
    result = 31 * result + Arrays.hashCode(values);
    result = 31 * result + Arrays.hashCode(basedOn);
    return result;
  }

  public BFGrid grid() {
    return features[0].row().grid();
  }

  @Override
  public double value(final BinarizedDataSet bds, final int pindex) {
    int index = 0;
    for (int i = 0; i < features.length; i++) {
      index <<= 1;
      if (features[i].value(bds.bins(features[i].findex)[pindex]))
        index++;
    }
    return values[index];
  }

  private class ObliviousGrad extends DiscontinuousTrans.Stub {
    @NotNull
    @Override
    public Vec leftTo(Vec x, Vec gradient) {
      final int numFeatures = x.dim();
      byte[] folds = new byte[numFeatures];
      grid.binarize(x, folds);

      final Vec x0 = VecTools.assign(new ArrayVec(numFeatures), x);
      final double value = ObliviousTree.this.value(x);

      for (int fIdx = 0; fIdx < numFeatures; fIdx++) {
        final BFGrid.BFRow row = grid.row(fIdx);
        final int leftIdx = folds[fIdx] - 1;

        if (leftIdx == -1) {
          gradient.adjust(fIdx, -1000);
        } else {
          final double leftBorder = row.borders[leftIdx];
          x0.set(fIdx, leftBorder);
          final double leftValue = ObliviousTree.this.value(x0);

          if (Math.abs(x.get(fIdx) - leftBorder) > 1e-7)
            gradient.adjust(fIdx, (value - leftValue) / (x.get(fIdx) - leftBorder));
          else
            gradient.adjust(fIdx, (value - leftValue));

          assert(!Double.isNaN(gradient.get(fIdx)));

          x0.set(fIdx, x.get(fIdx));
        }
      }

      return gradient;
    }

    @NotNull
    @Override
    public Vec rightTo(Vec x, Vec gradient) {
      final int numFeatures = x.dim();
      byte[] folds = new byte[numFeatures];
      grid.binarize(x, folds);

      final Vec x0 = VecTools.assign(new ArrayVec(numFeatures), x);
      final double value = ObliviousTree.this.value(x);

      for (int fIdx = 0; fIdx < numFeatures; fIdx++) {
        final BFGrid.BFRow row = grid.row(fIdx);
        final int rightIdx = folds[fIdx] + 1;
        if (rightIdx >= row.borders.length) {
          gradient.adjust(fIdx, 1000);
        } else {
          final double rightBorder = (row.borders[rightIdx] + row.borders[rightIdx - 1]) / 2.;
          x0.set(fIdx, rightBorder);

          final double rightValue = ObliviousTree.this.value(x0);
          if (Math.abs(x.get(fIdx) - rightBorder) > 1e-7)
            gradient.adjust(fIdx, (rightValue - value) / (rightBorder - x.get(fIdx)));
          else {
            gradient.adjust(fIdx, (rightValue - value));
          }
          assert(!Double.isNaN(gradient.get(fIdx)));

          x0.set(fIdx, x.get(fIdx));
        }
      }

      return gradient;
    }

    @Override
    public int xdim() {
      return ObliviousTree.this.dim();
    }

    @Override
    public int ydim() {
      return xdim();
    }
  }
}
