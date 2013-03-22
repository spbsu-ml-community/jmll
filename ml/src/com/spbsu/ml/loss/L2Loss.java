package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2Loss implements Oracle1 {
  private final Vec target;

  public L2Loss(Vec target) {
    this.target = target;
  }

  @Override
  public Vec gradient(Vec point) {
    Vec result = copy(point);
    scale(result, -1);
    append(result, target);
    return result;
  }

  public double value(Vec point) {
    return distance(target, point);
  }
}
