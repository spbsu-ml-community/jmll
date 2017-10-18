package com.expleague.ml.loss;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 13:23
 */
public abstract class DSSumFuncC1<Item> extends DSSumFunc<Item> implements FuncC1 {
  public DSSumFuncC1(DataSet<Item> ds) {
    super(ds);
  }

  @Override
  public abstract FuncC1 component(int index);

  @Override
  public Vec gradientTo(Vec x, Vec to) {
    final int length = length();
    VecTools.fill(to, 0.);
    for (int i = 0; i < length; i++){
      VecTools.append(to, component(i).gradient(x));
    }
    return to;
  }

  @Override
  public Vec gradient(Vec x) {
    return gradientTo(x, new ArrayVec(x.dim()));
  }

  @Override
  public Vec gradientRowTo(Vec x, Vec to, int index) {
    if (index != 0)
      throw new ArrayIndexOutOfBoundsException();
    return gradientTo(x, to);
  }
}
