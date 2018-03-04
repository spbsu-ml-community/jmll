package com.expleague.ml.methods;

import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;

public interface AnyOptimization<Loss extends TargetFunc, DSType extends DataSet<DSItem>, DSItem, Result> {
  Result fit(DSType learn, Loss loss);
}
