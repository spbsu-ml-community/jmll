package com.expleague.ml.models.hmm;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.vectors.*;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import org.jetbrains.annotations.NotNull;

import static java.lang.Math.exp;

public class HiddenMarkovModel<T> implements Computable<Seq<T>,Vec> {
  private final Alphabet<T> alpha;
  private final int statesCount;

  private final Vec betta;
  private final Vec Pi;
  private final Mx A;
  private final Mx inverseA;
  private final Mx B;

  public HiddenMarkovModel(Alphabet<T> alpha, int states, Vec betta) {
    this.betta = betta;
    this.alpha = alpha;
    statesCount = states;
    this.Pi = betta.sub(0, states);
    this.A = new VecBasedMx(states, betta.sub(states, states * states));
    this.B = new VecBasedMx(states, betta.sub((states + 1) * states, states * alpha.size()));
    this.inverseA = MxTools.transpose(A);
  }

  @Override
  public Vec compute(Seq<T> argument) {
    return new SingleValueVec(value(argument));
  }

  public double value(Seq<T> x) {
    final Mx distribs = forward(x);
    VecTools.scale(distribs, backward(x));
    double ll = 0;
    for (int t = 0; t < x.length(); t++) {
      final Vec states = distribs.row(t);
      VecTools.normalizeL1(states);
      ll += Math.log(VecTools.multiply(B.row(alpha.index(x, t)), states));
    }
    return exp(ll/x.length());
  }

  @NotNull
  public Mx forward(Seq<T> x) {
    final Mx forward = new VecBasedMx(statesCount, new ArrayVec(statesCount * x.length()));
    { // forward
      Vec prev = forward.row(0);
      VecTools.assign(prev, Pi);
      VecTools.scale(prev, B.row(alpha.index(x, 0)));
      VecTools.normalizeL1(prev);
      for (int i = 1; i < x.length(); i++) {
        final Vec next = forward.row(i);
        MxTools.multiplyTo(A, prev, next);
        VecTools.scale(next, B.row(alpha.index(x, i)));
        VecTools.normalizeL1(next);
        prev = next;
      }
    }
    return forward;
  }

  @NotNull
  public Mx backward(Seq<T> x) {
    final Mx backward = new VecBasedMx(statesCount, new ArrayVec(statesCount * x.length()));
    { // backward
      Vec prev = new ArrayVec(statesCount);
      VecTools.fill(prev, 1. / statesCount);
      for (int i = x.length() - 1; i >= 0; i--) {
        final Vec next = backward.row(i);
        final int index = alpha.index(x, i);
        MxTools.multiplyTo(inverseA, prev, next);
        VecTools.scale(next, B.row(index));
        VecTools.normalizeL1(next);
        prev = next;
      }
    }
    return backward;
  }

  public Vec betta() {
    return betta;
  }

  public int states() {
    return statesCount;
  }

  public Alphabet<T> alpha() {
    return alpha;
  }
}
