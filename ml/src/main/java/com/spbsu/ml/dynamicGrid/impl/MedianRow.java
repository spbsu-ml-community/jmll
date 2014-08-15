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
  private TIntArrayList borders = new TIntArrayList();
  private final ArrayList<BinaryFeatureImpl> bfs = new ArrayList<>();
  private final int levels;


  public MedianRow(DynamicGrid grid, double[] feature, int[] reverse, int origFIndex, int minSplits) {
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
    for (BinaryFeature bf : bfs) {
      bf.setActive(true);
    }
    addSplit();
  }

  public MedianRow(double[] feature, int[] reverse, int origFIndex) {
    this(null, feature, reverse, origFIndex, 1);
  }

  public MedianRow(double[] feature, int[] reverse, int origFIndex, int minSplits) {
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

  private static Random rand = new Random();


  private List<BinaryFeatureImpl> bestSplitsCache = new ArrayList<>();

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
      int ind = rand.nextInt(bestSplitsCache.size());
      BinaryFeatureImpl bf = bestSplitsCache.get(ind);
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


  private static double diffScore(int start, int end, int split, int n) {
    double diff = (end - start) * Math.log((end - start) * 1.0 / n)
            - (end - split) * Math.log((end - split) * 1.0 / n) - (split - start) * Math.log((split - start) * 1.0 / n);
    diff /= n;
    return diff;
  }

  private void updateCache() {
    double bestScore = 0;
    double diff = 0;
    int bestSplit = -1;
    TIntArrayList bestSplits = new TIntArrayList();
    for (int i = 0; i < borders.size(); ++i) {
      int start = i > 0 ? borders.get(i - 1) : 0;
      int end = borders.get(i);//.borderIndex;
      double median = feature[start + (end - start) / 2];
      int split = Math.abs(Arrays.binarySearch(feature, start, end, median));
      while (split > start && Math.abs(feature[split] - median) < eps) // look for first less then median value
        split--;
      if (Math.abs(feature[split] - median) > 1e-9) split++;

      final double scoreLeft = Math.log(end - split) + Math.log(split - start);
      if (split > start) {
        if (scoreLeft > bestScore) {
          bestScore = scoreLeft;
          diff = diffScore(start, end, split, feature.length);//(end - start) * Math.log(end - start) - (end - split) * Math.log(end - split) - (split - start) * Math.log(split - start);
//          diff = (end - start) * Math.log((end - start) * 1.0 / feature.length)
//                  - (end - split) * Math.log((end - split) * 1.0 / feature.length) - (split - start) * Math.log((split - start) * 1.0 / feature.length);
//          diff /= feature.length;
          bestSplit = split;
          bestSplits.clear();
          bestSplits.add(bestSplit);
        } else if (Math.abs(scoreLeft - bestScore) < 1e-8) {
          bestSplits.add(split);
        }
      }
      while (++split < end && Math.abs(feature[split] - median) < eps)
        ; // first after elements with such value

      final double scoreRight = Math.log(end - split) + Math.log(split - start);
      if (split < end) {
        if (scoreRight > bestScore) {
          bestScore = scoreRight;
          bestSplit = split;
//          diff = (end - start) * Math.log((end - start) * 1.0 / feature.length)
//                  - (end - split) * Math.log((end - split) * 1.0 / feature.length) - (split - start) * Math.log((split - start) * 1.0 / feature.length);
//          diff /= feature.length;
          diff = diffScore(start, end, split, feature.length);//(end - start) * Math.log(end - start) - (end - split) * Math.log(end - split) - (split - start) * Math.log(split - start);
          bestSplits.clear();
          bestSplits.add(bestSplit);
        } else if (Math.abs(scoreRight - bestScore) < 1e-8) {
          bestSplits.add(split);
        }
      }
    }


    bestSplitsCache.clear();

    for (int i = 0; i < bestSplits.size(); ++i) {
      bestSplit = bestSplits.get(i);
      borders.add(bestSplit);
      BinaryFeatureImpl newBF = new BinaryFeatureImpl(this, origFIndex, feature[bestSplit - 1], bestSplit);
      bestSplitsCache.add(newBF);
      newBF.setRegScore(diff);
    }

    int[] crcs = new int[bestSplitsCache.size()];
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
  public BinaryFeature bf(int binNo) {
    return bfs.get(binNo);
  }

  @Override
  public void setOwner(DynamicGrid grid) {
    this.grid = grid;
  }

  @Override
  public short bin(double value) {
    short index = 0;
    while (index < size() && value > bfs.get(index).condition)
      index++;
    return index;
  }


  @Override
  public double regularize(BinaryFeature bf) {
    double entropy = 0;
    double entropyWithoutFeature = 0;
    double left = 0;
    double leftWithoutFeature = 0;
    final int featureBin = bf.binNo();
    for (int bin = 0; bin < bfs.size(); ++bin) {
      BinaryFeatureImpl f = bfs.get(bin);
      double p = (f.borderIndex - left) / feature.length;
      entropy -= p * Math.log(p);
      left = f.borderIndex;
      if (bin != featureBin) {
        p = (f.borderIndex - leftWithoutFeature) / feature.length;
        entropyWithoutFeature -= p * Math.log(p);
        leftWithoutFeature = f.borderIndex;
      }
    }

    double p = (feature.length - left) / feature.length;
    entropy -= p * Math.log(p);
    p = (feature.length - leftWithoutFeature) / feature.length;
    entropyWithoutFeature -= p * Math.log(p);
    return entropy - entropyWithoutFeature;
  }

//    @Override
//    public void setActive(BinaryFeature feature) {
//        feature.setActive(true);
//    }
}
