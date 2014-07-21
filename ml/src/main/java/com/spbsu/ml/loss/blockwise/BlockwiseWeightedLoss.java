package com.spbsu.ml.loss.blockwise;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.BlockwiseFuncC1;

/**
* User: qdeee
* Date: 26.11.13
* Time: 9:54
*/

public class BlockwiseWeightedLoss<BasedOn extends BlockwiseFuncC1> extends BlockwiseFuncC1.Stub {
  private final BasedOn metric;
  private final int[] weights;

  public BlockwiseWeightedLoss(BasedOn metric, int[] weights) {
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
}
