package com.spbsu.ml.methods;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:14:38
 */
public interface Optimization<Loss extends Func, DSType extends DataSet<DSItem>, DSItem> {
  /**
   * @param learn X part of data set
   * @param loss is function of solution function results on each point of data set, loss.xdim() == solution.value(learn).dim() * f.dim()
   * @return function f = \arg \min_f Loss((f(learn_i))_1^m)
   */
  Computable<DSItem,Vec> fit(DSType learn, Loss loss);
}
