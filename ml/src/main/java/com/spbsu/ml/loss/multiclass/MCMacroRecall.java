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
public class MCMacroRecall extends Func.Stub implements TargetFunc {
  private final Vec target;
  private final DataSet<?> owner;
  private final int[] classLabels;

  public MCMacroRecall(final Vec target, DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
    this.classLabels = MCTools.getClassesLabels(target);
  }

  @Override
  public double value(Vec x) {
    TIntIntHashMap id2tp = new TIntIntHashMap();
    TIntIntHashMap id2fn = new TIntIntHashMap();
    for (int i = 0; i < target.dim(); i++) {
      int expected = (int) target.get(i);
      int actual = (int) x.get(i);
      if (actual == expected)
        id2tp.adjustOrPutValue(expected, 1, 1);
      else
        id2fn.adjustOrPutValue(expected, 1, 1);
    }

    double result = 0.;
    int nonEmpty = 0;
    for (int i = 0; i < classLabels.length; i++) {
      int cls = classLabels[i];
      int tp = id2tp.get(cls);
      int fn = id2fn.get(cls);
      if (tp + fn != 0) {
        nonEmpty++;
        result += tp / (0. + tp + fn);
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
