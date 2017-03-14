package com.spbsu.ml.loss;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.BlockedTargetFunc;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.TransC1;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 13:23
 */
public class DSSumFuncComposite<Item> extends DSSumFuncC1<Item> implements FuncC1 {
  private final BlockedTargetFunc tgt;
  private final Computable<Item, ? extends TransC1> decisionFactory;
  private final int dim;

  public DSSumFuncComposite(DataSet<Item> ds, BlockedTargetFunc tgt, Computable<Item, ? extends TransC1> decisionFactory) {
    super(ds);
    this.tgt = tgt;
    this.decisionFactory = decisionFactory;
    dim = decisionFactory.compute(ds.at(0)).xdim();
  }

  @Override
  public CompositeFunc component(int index) {
    return new CompositeFunc((FuncC1)tgt.block(index), decisionFactory.compute(ds.at(index)));
  }

  public Decision decision(final Vec x) {
    return new Decision(x);
  }

  @Override
  public Vec gradient(Vec x) {
    final Vec result = new ArrayVec(dim());
    final int length = length();
    for (int i = 0; i < length; i++){
      VecTools.append(result, component(i).gradient(x));
    }
    return result;
  }

  public int dim() {
    return dim;
  }

  public class Decision implements Computable<Item, Vec> {
    public final Vec x;

    public Decision(Vec x) {
      this.x = x;
    }

    @Override
    public Vec compute(Item argument) {
      final Trans compute = decisionFactory.compute(argument);
      return compute.trans(x);
    }
  }
}
