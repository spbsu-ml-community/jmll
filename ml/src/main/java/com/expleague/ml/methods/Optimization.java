package com.expleague.ml.methods;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;

import java.util.function.Function;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:14:38
 */
public interface Optimization<Loss extends TargetFunc, DSType extends DataSet<DSItem>, DSItem> {
  /**
   * @param learn X part of data set
   * @param loss is function of solution function results on each point of data set, loss.xdim() == solution.value(learn).dim() * f.dim()
   * @return function f = \arg \min_f Loss((f(learn_i))_1^m)
   */
  Function<DSItem,Vec> fit(DSType learn, Loss loss);
}
