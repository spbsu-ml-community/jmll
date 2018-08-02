package com.expleague.ml.func;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;

import java.util.function.Function;

@SuppressWarnings("unused")
public class LogitClassifier implements Function<Vec, Integer> {
  public int value(Vec x) {
    int v = VecTools.argmax(x);
    if (x.get(v) >=0)
      return v;
    return x.dim();
  }


  public Vec probs(Vec x) {
    final Vec result = new ArrayVec(x.dim() + 1);
    VecTools.assign(result.sub(0, x.dim()), x);
    VecTools.exp(result);
    VecTools.normalizeL1(result);
    return result;
  }

  @Override
  public Integer apply(Vec x) {
    return value(x);
  }
}
