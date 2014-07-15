package com.spbsu.ml.loss.multiclass;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.tools.MCTools;
import gnu.trove.map.hash.TIntIntHashMap;

/**
 * User: qdeee
 * Date: 09.04.14
 */
public class MCMacroPrecision extends Func.Stub implements TargetFunc {
  private final Vec target;
  private final DataSet<?> owner;
  private final int[] classLabels;

  public MCMacroPrecision(final Vec target, DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
    this.classLabels = MCTools.getClassesLabels(target);
  }

  @Override
  public double value(Vec x) {
    TIntIntHashMap id2tp = new TIntIntHashMap();
    TIntIntHashMap id2fp = new TIntIntHashMap();
    for (int i = 0; i < x.dim(); i++) {
      int expected = (int) target.get(i);
      int actual = (int) x.get(i);

      //skip unrecognized class
      if (actual == -1)
        continue;

      if (actual == expected)
        id2tp.adjustOrPutValue(actual, 1, 1);
      else
        id2fp.adjustOrPutValue(actual, 1, 1);
    }

    double result = 0.;
    int nonEmpty = 0;
    for (int i = 0; i < classLabels.length; i++) {
      int cls = classLabels[i];
      int tp = id2tp.get(cls);
      int fp = id2fp.get(cls);
      if (tp + fp != 0) {
        nonEmpty++;
        result += tp / (0. + tp + fp);
      }
    }
    return result / nonEmpty;
  }

  @Override
  public int dim() {
    return target.dim();
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }
}
