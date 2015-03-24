package com.spbsu.ml.func;

import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;

import static com.spbsu.commons.math.vectors.VecTools.append;

/**
 * User: solar
 * Date: 01.03.11
 * Time: 22:30
 */
public class NormalizedLinear extends Linear {
  private final double avg;
  private final MxTools.NormalizationProperties props;

  public NormalizedLinear(final double avg, final Vec weights, final MxTools.NormalizationProperties props) {
    super(weights);
    this.avg = avg;
    this.props = props;
  }

  @Override
  public double value(final Vec point) {
    final Vec x = MxTools.multiply(props.xTrans, point);
    append(x, props.xMean);
    return super.value(point) + avg;
  }
}
