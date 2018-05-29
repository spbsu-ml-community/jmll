package com.expleague.ml.methods.seq;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.ml.methods.seq.param.BettaParametrization;
import com.expleague.ml.methods.seq.param.WeightParametrization;

import java.util.function.Function;

class PNFAModel<Type> implements Function<Seq<Type>, Vec> {
  private Vec params;
  private int stateCount;
  private int stateDim;
  private double addToDiag;
  private double lambda;
  private Alphabet<Type> alphabet;
  private BettaParametrization bettaParametrization;
  private WeightParametrization weightParametrization;

  public PNFAModel(Vec params, int stateCount, int stateDim, double addToDiag, double lambda, Alphabet<Type> alpha, BettaParametrization bettaParametrization, WeightParametrization weightParametrization) {
    this.params = params;
    this.stateCount = stateCount;
    this.stateDim = stateDim;
    this.addToDiag = addToDiag;
    this.lambda = lambda;
    this.alphabet = alpha;
    this.bettaParametrization = bettaParametrization;
    this.weightParametrization = weightParametrization;
  }

  @Override
  public Vec apply(Seq<Type> seq) {
    PNFAItemVecRegression regression = new PNFAItemVecRegression(
        alphabet.reindex(seq),
        Vec.EMPTY,
        stateCount,
        alphabet.size(),
        stateDim,
        bettaParametrization,
        weightParametrization
    );
    return regression.vecValue(params);
  }

  public Vec getParams() {
    return params;
  }

  public void setParams(Vec params) {
    this.params = params;
  }

  public int getStateCount() {
    return stateCount;
  }

  public void setStateCount(int stateCount) {
    this.stateCount = stateCount;
  }

  public int getStateDim() {
    return stateDim;
  }

  public void setStateDim(int stateDim) {
    this.stateDim = stateDim;
  }

  public double getAddToDiag() {
    return addToDiag;
  }

  public void setAddToDiag(double addToDiag) {
    this.addToDiag = addToDiag;
  }

  public double getLambda() {
    return lambda;
  }

  public void setLambda(double lambda) {
    this.lambda = lambda;
  }
}
