package com.spbsu.ml.loss.multiclass;


import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.tools.MCTools;
import gnu.trove.map.hash.TIntIntHashMap;

/**
 * User: qdeee
 * Date: 09.04.14
 */
public class MCMacroRecall extends Func.Stub implements ClassicMulticlassLoss {
  private final IntSeq target;
  private final DataSet<?> owner;
  private final int[] classLabels;

  public MCMacroRecall(final IntSeq target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
    this.classLabels = MCTools.getClassesLabels(target);
  }

  public MCMacroRecall(final Vec target, final DataSet<?> owner) {
    final int[] intTarget = new int[target.length()];
    final VecIterator iter = target.nonZeroes();
    while (iter.advance()) {
      intTarget[iter.index()] = (int) iter.value();
    }
    this.target = new IntSeq(intTarget);
    this.owner = owner;
    this.classLabels = MCTools.getClassLabels(target);
  }

  @Override
  public double value(final Vec x) {
    final TIntIntHashMap id2tp = new TIntIntHashMap();
    final TIntIntHashMap id2fn = new TIntIntHashMap();
    for (int i = 0; i < target.length(); i++) {
      final int expected = target.at(i);
      final int actual = (int) x.get(i);
      if (actual == expected)
        id2tp.adjustOrPutValue(expected, 1, 1);
      else
        id2fn.adjustOrPutValue(expected, 1, 1);
    }

    double result = 0.;
    int nonEmpty = 0;
    for (int i = 0; i < classLabels.length; i++) {
      final int cls = classLabels[i];
      final int tp = id2tp.get(cls);
      final int fn = id2fn.get(cls);
      if (tp + fn != 0) {
        nonEmpty++;
        result += tp / (0. + tp + fn);
      }
    }
    return result / nonEmpty;
  }

  @Override
  public int dim() {
    return target.length();
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }

  @Override
  public IntSeq labels() {
    return target;
  }
}
