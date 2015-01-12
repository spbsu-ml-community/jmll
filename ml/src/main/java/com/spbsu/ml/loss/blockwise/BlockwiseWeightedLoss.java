package com.spbsu.ml.loss.blockwise;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.BlockwiseFuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

/**
* User: qdeee
* Date: 26.11.13
* Time: 9:54
*/

public class BlockwiseWeightedLoss<BasedOn extends BlockwiseFuncC1 & TargetFunc> extends BlockwiseFuncC1.Stub implements TargetFunc{
  private final BasedOn metric;
  private final int[] weights;

  public BlockwiseWeightedLoss(final BasedOn metric, final int[] weights) {
    if (metric.dim() / metric.blockSize() != weights.length)
      throw new IllegalArgumentException("weights.length must be equal to blocks count");
    this.metric = metric;
    this.weights = weights;
  }

  @Override
  public int dim() {
    return metric.xdim();
  }

  public double weight(final int index) {
    return weights[index];
  }

  public BasedOn base() {
    return metric;
  }

  @Override
  public void gradient(final Vec pointBlock, final Vec result, final int index) {
    if (weights[index] > 0) {
      metric.gradient(pointBlock, result, index);
      VecTools.scale(result, weights[index]);
    }
  }

  @Override
  public double value(final Vec pointBlock, final int index) {
    return weights[index] > 0 ? weights[index] * metric.value(pointBlock, index) : 0;
  }

  @Override
  public double transformResultValue(final double value) {
    return metric.transformResultValue(value);
  }

  @Override
  public int blockSize() {
    return metric.blockSize();
  }

  @Override
  public DataSet<?> owner() {
    return metric.owner();
  }
}
