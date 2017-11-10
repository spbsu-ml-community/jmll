package com.expleague.ml.loss;

import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.BlockedTargetFunc;
import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.TransC1;
import com.expleague.ml.data.set.DataSet;

import java.util.function.Function;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 13:23
 */
public class DSSumFuncComposite<Item> extends DSSumFuncC1<Item> implements FuncC1 {
  private final BlockedTargetFunc tgt;
  private final Function<Item, ? extends TransC1> decisionFactory;
  private final int dim;

  public DSSumFuncComposite(DataSet<Item> ds, BlockedTargetFunc tgt, Function<Item, ? extends TransC1> decisionFactory) {
    super(ds);
    this.tgt = tgt;
    this.decisionFactory = decisionFactory;
    dim = decisionFactory.apply(ds.at(0)).xdim();
  }

  @Override
  public CompositeFunc component(int index) {
    return new CompositeFunc((FuncC1)tgt.block(index), decisionFactory.apply(ds.at(index)));
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

  public class Decision implements Function<Item, Vec> {
    public final Vec x;

    public Decision(Vec x) {
      this.x = x;
    }

    @Override
    public Vec apply(Item argument) {
      final Trans compute = decisionFactory.apply(argument);
      return compute.trans(x);
    }
  }
}
