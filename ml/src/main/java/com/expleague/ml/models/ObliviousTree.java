package com.expleague.ml.models;

import com.expleague.commons.math.Trans;
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
  public Trans gradient() {
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

  @Nullable
  public static ObliviousTree removeFeatures(@NotNull final ObliviousTree tree, final int ... indexes) {
    final int[] sortedIndexes = Arrays.copyOf(indexes, indexes.length);
    Arrays.sort(sortedIndexes);
    return removeFeaturesNoSort(tree, sortedIndexes);
  }

  @Nullable
  public static ObliviousTree removeFeaturesNoSort(@Nullable final ObliviousTree tree, final int ... indexes) {
    if (indexes.length == 0)
      return tree;
    if (tree == null)
      return null;

    for (int i = 0; i < tree.features.length; i++) {
      final BFGrid.BinaryFeature bf = tree.features[i];
      final int findex = Arrays.binarySearch(indexes, bf.findex);
      if (findex >= 0) {
        return removeFeaturesNoSort(removeBF(tree, bf), indexes);
      }
    }
    return tree;
  }

  @Nullable
  private static ObliviousTree removeBF(@NotNull final ObliviousTree tree, @NotNull final BFGrid.BinaryFeature bf) {
    final double[] values = new double[tree.values.length >> 1];
    final double[] basedOn = new double[tree.basedOn.length >> 1];
    final BFGrid.BinaryFeature[] features = new BFGrid.BinaryFeature[tree.features.length - 1];

    int findex = -1;
    int idx = 0;
    for (int i = 0; i < tree.features.length; i++) {
      if (!tree.features[i].equals(bf)) {
        features[idx++] = tree.features[i];
      } else {
        assert findex == -1;
        findex = i;
      }
    }

    assert findex != -1;

    final int mask = 1 << (tree.features.length - findex - 1);

    final int border = (1 << tree.features.length) - 1;

    int heighMask = (1 << features.length) - 1;
    if (mask < heighMask) {
      heighMask = (heighMask - (mask << 1) - 1) & mask;
    }
    final int lowMask = border ^ heighMask;

    for (int i = 0; i < values.length; i++) {
      final int left = (2 * (i & heighMask) + (i & lowMask)) & border;
      final int right = (left + mask) & border;
      final double leftBase = tree.basedOn[left];
      final double rightBase = tree.basedOn[right];
      assert leftBase > 0 && rightBase > 0;
      double lk = leftBase / (leftBase + rightBase);
      basedOn[i] = leftBase + rightBase;
      if (basedOn[i] == 0)
        lk = .5;
      values[i] = tree.values[left] * lk + tree.values[right] * (1 - lk);
    }

    return features.length > 0 ? new ObliviousTree(Arrays.asList(features), values, basedOn) : null;
  }

  private class ObliviousGrad extends Trans.Stub {
    @Override
    public Vec transTo(Vec x, Vec gradient) {
      final int numFeatures = x.dim();
      byte[] folds = new byte[numFeatures];
      grid.binarize(x, folds);

      final double[] leftBorders = new double[numFeatures];
      final double[] rightBorders = new double[numFeatures];

      for (int i = 0; i < numFeatures; i++) {
        final BFGrid.BFRow row = grid.row(i);
        final int leftIdx = folds[i] - 1;
        final int rightIdx = folds[i] + 1;

        leftBorders[i] = leftIdx == -1 ? x.get(i) : row.borders[leftIdx];
        rightBorders[i] = rightIdx >= row.borders.length ? x.get(i)
            : (row.borders[rightIdx] + row.borders[rightIdx - 1]) / 2.;

//        System.out.println(row.bfStart + " " + row.bfEnd);
//        System.out.println(leftBorders[i] + " " + rightBorders[i]);
//        assert(leftBorders[i] <= x.get(i) && x.get(i) <= rightBorders[i]);
      }

      final Vec leftGrad = gradient.sub(0, numFeatures);
      final Vec rightGrad = gradient.sub(numFeatures, numFeatures);

      final Vec x0 = VecTools.assign(new ArrayVec(numFeatures), x);
      final double value = ObliviousTree.this.value(x);

      for (int fIdx = 0; fIdx < numFeatures; fIdx++) {
        x0.set(fIdx, leftBorders[fIdx]);
        final double leftValue = ObliviousTree.this.value(x0);
        x0.set(fIdx, rightBorders[fIdx]);
        final double rightValue = ObliviousTree.this.value(x0);

        leftGrad.adjust(fIdx, value - leftValue);
        rightGrad.adjust(fIdx, rightValue - value);

        x0.set(fIdx, x.get(fIdx));
      }

      return gradient;
    }

    @Override
    public int xdim() {
      return ObliviousTree.this.dim();
    }

    @Override
    public int ydim() {
      return xdim() * 2;
    }
  }
}
