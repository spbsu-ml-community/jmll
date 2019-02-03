package com.expleague.ml.methods.seq.framework;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.Seq;

import java.util.Map;
import java.util.function.Function;

/**
 * Created by hrundelb on 02.02.19.
 */
public class DFAModel<Type> implements Function<Seq<Type>, Vec> {

  private final int stateCount;
  private final Map<Type, Mx> weightsMap;

  public DFAModel(int stateCount, Map<Type, Mx> weightsMap) {
    this.stateCount = stateCount;
    this.weightsMap = weightsMap;
  }

  public Mx getWeightsMx(Type t) {
    return weightsMap.get(t);
  }

  @Override
  public Vec apply(Seq<Type> seq) {
    Vec[] distribution = new Vec[] {new ArrayVec(stateCount), new ArrayVec(stateCount)};
    VecTools.fill(distribution[0], 1.0 / stateCount);
    for (int i = 0; i < seq.length(); i++) {
      Mx weightMx = getWeightsMx(seq.at(i));
      MxTools.multiplyTo(weightMx, distribution[i % 2], distribution[(i + 1) % 2]);
    }

    return distribution[seq.length() % 2];
  }
}
