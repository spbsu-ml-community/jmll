package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.impl.Bootstrap;

import static java.lang.Math.log;

/**
* User: solar
* Date: 10.09.13
* Time: 12:14
*/
public class LLClassificationLeaf implements BFLeaf {
  private final Vec point;
  private final Vec target;
  private final Vec weight;
  private final BinarizedDataSet ds;
  private int[] indices;
  protected final LLCounter[] counters;
  LLCounter total = new LLCounter();
  private final BFGrid.BFRow[] rows;

  public LLClassificationLeaf(BinarizedDataSet ds, Vec point, Vec target, Vec weight) {
    this(ds, point, target, weight, ds.original() instanceof Bootstrap ? ((Bootstrap) ds.original()).order() : ArrayTools.sequence(0, ds.original().power()));
  }

  public LLClassificationLeaf(BinarizedDataSet ds, int[] points, LLCounter right, LLClassificationLeaf bro) {
    this.ds = ds;
    point = bro.point;
    target = bro.target;
    weight = bro.weight;
    this.indices = points;
    total = right;
    counters = new LLCounter[ds.grid().size() + ds.grid().rows()];
    for (int i = 0; i < counters.length; i++) {
      counters[i] = new LLCounter();
    }
    rows = ds.grid().allRows();
  }

  public LLClassificationLeaf(BinarizedDataSet ds, Vec point, Vec target, Vec weight, int[] pointIndices) {
    this.ds = ds;
    this.point = point;
    this.indices = pointIndices;
    this.target = target;
    this.weight = weight;
    for (int i = 0; i < this.indices.length; i++) {
      final int index = indices[i];
      total.found(this.point.get(index), target.get(index), weight.get(index));
    }
    counters = new LLCounter[ds.grid().size() + ds.grid().rows()];
    for (int i = 0; i < counters.length; i++) {
      counters[i] = new LLCounter();
    }
    rows = ds.grid().allRows();
    ds.aggregate(this, target, this.point, this.indices);
  }

  @Override
  public void append(int feature, byte bin, double target, double current, double weight) {
    counter(feature, bin).found(current, target, weight);
  }

  public LLCounter counter(int feature, byte bin) {
    final BFGrid.BFRow row = rows[feature];
    return counters[1 + feature + (bin > 0 ? row.bf(bin - 1).bfIndex : row.bfStart - 1)];
  }

  @Override
  public int score(final double[] likelihoods) {
    for (int f = 0; f < rows.length; f++) {
      final BFGrid.BFRow row = rows[f];
      LLCounter left = new LLCounter();
      LLCounter right = new LLCounter();
      right.add(total);

      for (int b = 0; b < row.size(); b++) {
        left.add(counter(f, (byte) b));
        right.sub(counter(f, (byte) b));
        likelihoods[row.bfStart + b] = score(left) + score(right);
      }
    }
    return ArrayTools.max(likelihoods);
  }

  @Override
  public int size() {
    return indices.length;
  }

  private double score(LLCounter counter) {
    return counter.score()/log(counter.size() + 2);
  }

  /** Splits this leaf into two right side is returned */
  @Override
  public LLClassificationLeaf split(BFGrid.BinaryFeature feature) {
    LLCounter left = new LLCounter();
    LLCounter right = new LLCounter();
    right.add(total);

    for (int b = 0; b <= feature.binNo; b++) {
      left.add(counter(feature.findex, (byte) b));
      right.sub(counter(feature.findex, (byte) b));
    }

    final int[] leftPoints = new int[left.size()];
    final int[] rightPoints = new int[right.size()];
    final LLClassificationLeaf brother = new LLClassificationLeaf(ds, rightPoints, right, this);

    {
      int leftIndex = 0;
      int rightIndex = 0;
      byte[] bins = ds.bins(feature.findex);
      byte splitBin = (byte)feature.binNo;
      for (int i = 0; i < indices.length; i++) {
        final int point = indices[i];
        if (bins[point] > splitBin)
          rightPoints[rightIndex++] = point;
        else
          leftPoints[leftIndex++] = point;
      }
      ds.aggregate(brother, target, point, rightPoints);
    }
    for (int i = 0; i < counters.length; i++) {
      counters[i].sub(brother.counters[i]);
    }
    indices = leftPoints;
    total = left;
    return brother;
  }

  @Override
  public double alpha() {
    return total.alpha();
  }
}
