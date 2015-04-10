package com.spbsu.ml.data.impl;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.stats.OrderByFeature;


/**
 * Created by noxoomo on 02/04/15.
 */
public class RankedDataSet {
  private final DataSet base;
  private final float[][] ranks;
  final OrderByFeature byFeature;

  public RankedDataSet(DataSet base) {
    this.base = base;
    byFeature = base.cache().cache(OrderByFeature.class, DataSet.class);
    final Mx data = ((VecDataSet) base).data();
    this.ranks = new float[data.columns()][data.rows()];
    for (int feature=0; feature < ranks.length;++feature) {
      int[] order = byFeature.orderBy(feature).direct();
      ranks[feature] = rank(data, feature, order);
    }
  }


  private float[] rank(final double[] sortedValues) {
    final float[] ranks = new float[sortedValues.length];
    for (int i = 0; i < sortedValues.length; ++i) {
      int j = i + 1;
      while (j < sortedValues.length && Math.abs(sortedValues[j] - sortedValues[j - 1]) < 1e-9) ++j;
      final float rk = i + 0.5f * (j - i);
      for (; i < j; ++i) {
        ranks[i] = rk;

      }
      --i;
    }
    return ranks;
  }

  private float[] rank(final Mx data,int feature, int[] order) {
    final float[] ranks = new float[order.length];
    for (int i = 0; i < order.length; ++i) {
      int j = i + 1;
      while (j < order.length && Math.abs(data.get(order[j],feature)-data.get(order[j-1],feature)) < 1e-9) ++j;
      final float rk = i + 0.5f * (j - i);
      for (; i < j; ++i) {
        ranks[order[i]] = rk;
      }
      --i;
    }
    return ranks;
  }

  public float[] feature(int fIndex) {
    return ranks[fIndex];
  }


  float rank(int fIndex, float condition) {
    final Mx data = ((VecDataSet) base).data();
    return upperBound(data.col(fIndex), byFeature.orderBy(fIndex).direct(), condition);
  }


  //java version doesn't guarantee, that we'll find last entry
  //should return first index, that greater than key
  private int upperBound(final double[] arr, final double key) {
    int left = 0;
    int right = arr.length;
    while (right - left > 1) {
      final int mid = (left + right) >>> 1;
      final double midVal = arr[mid];
      if (midVal <= key)
        left = mid;
      else
        right = mid;
    }
    return right;
  }


  //java version doesn't guarantee, that we'll find last entry
  //should return first index, that greater than key
  private int upperBound(final Vec feature, final int[] map, final double key) {
    int left = 0;
    int right = map.length;
    while (right - left > 1) {
      final int mid = (left + right) >>> 1;
      final double midVal = feature.get(map[mid]);
      if (midVal <= key)
        left = mid;
      else
        right = mid;
    }
    return right;
  }


}
