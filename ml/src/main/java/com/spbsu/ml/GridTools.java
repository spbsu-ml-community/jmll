package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.stats.OrderByFeature;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.hash.TIntHashSet;

import java.util.Arrays;

/**
 * User: solar
 * Date: 27.07.12
 * Time: 17:42
 */
public class GridTools {
    public static BFGrid medianGrid(final DataSet<Vec> ds, final int binFactor) {
        final int dim = ((VecDataSet) ds).xdim();
        final BFGrid.BFRow[] rows = new BFGrid.BFRow[dim];
        final TIntHashSet known = new TIntHashSet();
        final OrderByFeature byFeature = ds.cache().cache(OrderByFeature.class, DataSet.class);
        final TIntArrayList borders = new TIntArrayList();
        int bfCount = 0;

        final double[] feature = new double[ds.length()];
        for (int f = 0; f < dim; f++) {
            borders.clear();
            borders.add(ds.length());
            final ArrayPermutation permutation = byFeature.orderBy(f);
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
            while (borders.size() < binFactor + 1) {
                double bestScore = 0;
                int bestSplit = -1;
                for (int i = 0; i < borders.size(); i++) {
                    final int start = i > 0 ? borders.get(i - 1) : 0;
                    final int end = borders.get(i);
                    final double median = feature[start + (end - start) / 2];
                    int split = Math.abs(Arrays.binarySearch(feature, start, end, median));

                    while (split > 0 && Math.abs(feature[split] - median) < 1e-9) // look for first less then median value
                        split--;
                    if (Math.abs(feature[split] - median) > 1e-9) split++;
                    final double scoreLeft = Math.log(end - split) + Math.log(split - start);
                    if (split > 0 && scoreLeft > bestScore) {
                        bestScore = scoreLeft;
                        bestSplit = split;
                    }
                    while (++split < end && Math.abs(feature[split] - median) < 1e-9)
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
}
