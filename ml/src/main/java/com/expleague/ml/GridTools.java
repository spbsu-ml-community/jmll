package com.expleague.ml;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.TDoubleSet;
import gnu.trove.set.hash.TDoubleHashSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.Arrays;

/**
 * User: solar
 * Date: 27.07.12
 * Time: 17:42
 */
public class GridTools {

  public static TIntArrayList greedyLogSumBorders(final double[] sortedFeature,
                                                  final int binFactor) {
    final TIntArrayList borders = new TIntArrayList();
    borders.add(sortedFeature.length);

    while (borders.size() < binFactor + 1) {
      double bestScore = 0;
      int bestSplit = -1;
      for (int i = 0; i < borders.size(); i++) {
        final int start = i > 0 ? borders.get(i - 1) : 0;
        final int end = borders.get(i);
        final double median = sortedFeature[start + (end - start) / 2];
        int split = Math.abs(Arrays.binarySearch(sortedFeature, start, end, median));

        while (split > 0 && Math.abs(sortedFeature[split] - median) < 1e-9) // look for first less then median value
          split--;
        if (Math.abs(sortedFeature[split] - median) > 1e-9) split++;
        final double scoreLeft = Math.log(end - split) + Math.log(split - start);
        if (split > 0 && scoreLeft > bestScore) {
          bestScore = scoreLeft;
          bestSplit = split;
        }
        while (++split < end && Math.abs(sortedFeature[split] - median) < 1e-9)
          ; // first after elements with such value
        final double scoreRight = Math.log(end - split) + Math.log(split - start);
        if (split < end && scoreRight > bestScore) {
          bestScore = scoreRight;
          bestSplit = split;
        }
      }

      if (bestSplit < 0)
        break;
      borders.add(bestSplit);
      borders.sort();
    }
    return borders;
  }

  public static TDoubleSet uniqueValuesSet(final Vec src) {
    TDoubleHashSet values = new TDoubleHashSet();
    for (int i = 0; i < src.dim(); ++i) {
      values.add(src.get(i));
    }
    return values;
  }

  public static double[] sortUnique(final Vec vec) {
    TDoubleSet values = uniqueValuesSet(vec);
    final double[] result = values.toArray();
    Arrays.sort(result);
    return result;
  }

  public static int uniqueValues(final Vec vec) {
    return uniqueValuesSet(vec).size();
  }

  public static BFGrid medianGrid(final VecDataSet ds, final int binFactor) {
    final int dim = ds.xdim();
    final BFGrid.BFRow[] rows = new BFGrid.BFRow[dim];
    final TIntHashSet known = new TIntHashSet();
    int bfCount = 0;

    final double[] feature = new double[ds.length()];
    for (int f = 0; f < dim; f++) {
      final ArrayPermutation permutation = new ArrayPermutation(ds.order(f));
      final int[] order = permutation.direct();
      final int[] reverse = permutation.reverse();
      boolean haveDiffrentElements = false;
      for (int i = 1; i < order.length; i++)
        if (order[i] != order[0])
          haveDiffrentElements = true;
      if (!haveDiffrentElements)
        continue;
      for (int i = 0; i < feature.length; i++)
        feature[i] = ds.at(order[i]).get(f);
      final TIntArrayList borders = greedyLogSumBorders(feature, binFactor);
      final TDoubleArrayList dborders = new TDoubleArrayList();
      final TIntArrayList sizes = new TIntArrayList();
      { // drop existing
        final int[] crcs = new int[borders.size()];
        for (int i = 0; i < ds.length(); i++) { // unordered index
          final int orderedIndex = reverse[i];
          for (int b = 0; b < borders.size() && orderedIndex >= borders.get(b); b++) {
            crcs[b] = (crcs[b] * 31) + (i + 1);
          }
        }
        for (int b = 0; b < borders.size() - 1; b++) {
          if (known.contains(crcs[b]))
            continue;
          known.add(crcs[b]);
          dborders.add(feature[borders.get(b) - 1]);
          sizes.add(borders.get(b));
        }
      }
      rows[f] = new BFGrid.BFRow(bfCount, f, dborders.toArray(), sizes.toArray());

      bfCount += dborders.size();
    }
    return new BFGrid(rows);
  }

  public static double[] quantileBorders(final double[] values, final int binFactor) {
    final double[] borders = new double[binFactor];
    for (int i = 0; i < borders.length; ++i) {
      borders[i] = values[(int) (1.0 * (i + 1.0) * values.length  / (borders.length + 0.1))];
    }
    return sortUnique(new ArrayVec(borders));
  }
}
