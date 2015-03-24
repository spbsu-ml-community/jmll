package com.spbsu.ml.dynamicGrid.impl;

import com.spbsu.ml.dynamicGrid.interfaces.BinaryFeature;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicRow;
import gnu.trove.list.array.TIntArrayList;

import java.util.*;

/**
 * Created by noxoomo on 23/07/14.
 */
public class MedianRow implements DynamicRow {
  private final static double eps = 1e-9;
  private final int origFIndex;
  private DynamicGrid grid = null;
  private final double[] feature;
  private final int[] reverse;
  private final TIntArrayList borders = new TIntArrayList();
  private final ArrayList<BinaryFeatureImpl> bfs = new ArrayList<>();
  private final int levels;

  public MedianRow(final DynamicGrid grid, final double[] feature, final int[] reverse, final int origFIndex, final int minSplits) {
    this.origFIndex = origFIndex;
    this.feature = feature;
    this.grid = grid;
    this.reverse = reverse;
    int lvl = 0;
    for (int i = 1; i < feature.length; ++i)
      if (feature[i] != feature[i - 1])
        ++lvl;
    this.levels = lvl;
    borders.add(feature.length);
    for (int i = 0; i < minSplits; ++i)
      addSplit();
    for (final BinaryFeature bf : bfs) {
      bf.setActive(true);
    }
    addSplit();
  }

  public MedianRow(final double[] feature, final int[] reverse, final int origFIndex) {
    this(null, feature, reverse, origFIndex, 1);
  }

  public MedianRow(final double[] feature, final int[] reverse, final int origFIndex, final int minSplits) {
    this(null, feature, reverse, origFIndex, minSplits);
  }

  public ArrayList<BinaryFeatureImpl> features() {
    return bfs;
  }

  @Override
  public int origFIndex() {
    return origFIndex;
  }


  @Override
  public int size() {
    return bfs.size();
  }

  @Override
  public DynamicGrid grid() {
    return grid;
  }

  private static final Random rand = new Random();


  private final List<BinaryFeatureImpl> bestSplitsCache = new ArrayList<>();

  @Override
  public boolean addSplit() {
    if (bfs.size() >= levels + 1)
      return false;
    if (bestSplitsCache.size() == 0) {
      updateCache();
    }
    while (!addFromCache()) {
      updateCache();
      if (bestSplitsCache.size() == 0)
        return false;
    }
    return true;
  }

  private boolean addFromCache() {
    while (bestSplitsCache.size() > 0) {
      final int ind = rand.nextInt(bestSplitsCache.size());
      final BinaryFeatureImpl bf = bestSplitsCache.get(ind);
      bestSplitsCache.remove(ind);
      if (grid.isKnown(bf.gridHash)) {
        continue;
      }
      bfs.add(bf);
      grid.setKnown(bf.gridHash);
      Collections.sort(bfs, BinaryFeatureImpl.borderComparator);
      for (int i = 0; i < bfs.size(); ++i) {
        bfs.get(i).setBinNo(i);
      }
//      bf.setRegScore(bf.row().regularize(bf));
      return true;
    }
    return false;
  }


  private static double diffScore(final int start, final int end, final int split, final int n) {
    double diff = (end - start) * Math.log((end - start) * 1.0 / n)
            - (end - split) * Math.log((end - split) * 1.0 / n) - (split - start) * Math.log((split - start) * 1.0 / n);
//    diff /= n;
    diff /= n;
    return diff;
  }

  private void updateCache() {
    double bestScore = 0;
    double diff = 0;
    int bestSplit = -1;
    final TIntArrayList bestSplits = new TIntArrayList();
    for (int i = 0; i < borders.size(); ++i) {
      final int start = i > 0 ? borders.get(i - 1) : 0;
      final int end = borders.get(i);//.borderIndex;
      final double median = feature[start + (end - start) / 2];
      int split = Math.abs(Arrays.binarySearch(feature, start, end, median));
      while (split > start && Math.abs(feature[split] - median) < eps) // look for first less then median value
        split--;
      if (Math.abs(feature[split] - median) > eps) split++;

      final double scoreLeft = split > start ? Math.log(end - split) + Math.log(split - start) : 0;
      final int lb = split;
      while (++split < end && Math.abs(feature[split] - median) < eps)
        ; // first after elements with such value
      final double scoreRight = split < end ? Math.log(end - split) + Math.log(split - start) : 0;
      final int ub = split;
      split = scoreLeft < scoreRight ? ub : lb;
      final double score = scoreLeft < scoreRight ? scoreRight : scoreLeft;

      if (split > start && split < end) {
        if (score > bestScore + eps) {
          bestScore = score;
          bestSplit = split;
          diff = diffScore(start, end, split, feature.length);//(end - start) * Math.log(end - start) - (end - split) * Math.log(end - split) - (split - start) * Math.log(split - start);
          bestSplits.clear();
          bestSplits.add(bestSplit);
        } else if (Math.abs(score - bestScore) < eps) {
          bestSplits.add(split);
        }
      }
    }
    bestSplitsCache.clear();

    for (int i = 0; i < bestSplits.size(); ++i) {
      bestSplit = bestSplits.get(i);
      borders.add(bestSplit);
      final BinaryFeatureImpl newBF = new BinaryFeatureImpl(this, origFIndex, feature[bestSplit - 1], bestSplit);
      bestSplitsCache.add(newBF);
      newBF.setRegScore(diff);
    }

    final int[] crcs = new int[bestSplitsCache.size()];
    for (int i = 0; i < feature.length; i++) { // unordered index
      final int orderedIndex = reverse[i];
      for (int b = 0; b < bestSplitsCache.size() && orderedIndex >= bestSplitsCache.get(b).borderIndex; b++) {
        crcs[b] = (crcs[b] * 31) + (i + 1);
      }
    }
    for (int i = 0; i < bestSplitsCache.size(); ++i) {
      bestSplitsCache.get(i).gridHash = crcs[i];
    }
    borders.sort();
  }


  @Override
  public boolean empty() {
    return size() == 0;
  }

  @Override
  public BinaryFeature bf(final int binNo) {
    return bfs.get(binNo);
  }

  @Override
  public void setOwner(final DynamicGrid grid) {
    this.grid = grid;
  }

  @Override
  public short bin(final double value) {
    short index = 0;
    while (index < size() && value > bfs.get(index).condition)
      index++;
    return index;
  }


  @Override
  public double regularize(final BinaryFeature bf) {
    return bf.regularization();
//    double entropy = 0;
//    double entropyWithoutFeature = 0;
//    double left = 0;
//    double leftWithoutFeature = 0;
//    final int featureBin = bf.binNo();
//    for (int bin = 0; bin < bfs.size(); ++bin) {
//      BinaryFeatureImpl f = bfs.get(bin);
//      double p = (f.borderIndex - left) / feature.length;
//      entropy -= p * Math.log(p);
//      left = f.borderIndex;
//      if (bin != featureBin) {
//        p = (f.borderIndex - leftWithoutFeature) / feature.length;
//        entropyWithoutFeature -= p * Math.log(p);
//        leftWithoutFeature = f.borderIndex;
//      }
//    }
//
//    double p = (feature.length - left) / feature.length;
//    entropy -= p * Math.log(p);
//    p = (feature.length - leftWithoutFeature) / feature.length;
//    entropyWithoutFeature -= p * Math.log(p);
//    return entropy - entropyWithoutFeature;
  }

//    @Override
//    public void setActive(BinaryFeature feature) {
//        feature.setActive(true);
//    }
}
