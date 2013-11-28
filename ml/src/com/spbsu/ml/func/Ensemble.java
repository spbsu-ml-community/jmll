package com.spbsu.ml.func;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.CompositeTrans;
import com.spbsu.ml.Trans;

import java.util.List;

/**
 * User: solar
 * Date: 26.11.13
 * Time: 9:41
 */
public class Ensemble extends CompositeTrans<Linear, TransJoin> {
  public Ensemble(Trans[] models, Vec weights) {
    super(new Linear(weights), new TransJoin(models));
  }

  public Ensemble(List<Trans> weakModels, double step) {
    this(weakModels.toArray(new Trans[weakModels.size()]), VecTools.fill(new ArrayVec(weakModels.size()), step));
  }

  public double wlast() {
    return f.weights.get(size() - 1);
  }

  public Trans last() {
    return g.dirs[size() - 1];
  }

  public int size() {
    return f.dim();
  }
}
