package com.expleague.ml.bayesianEstimation;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;

import java.util.List;


public interface ConjugateBayesianEstimator<U extends RandomVariable<U>> {

  U clone(final U dist);

  U improperPrior();

  //parameter space
  int dim();

  RandomVec<U> update(final int idx,
                      final double observation,
                      final RandomVec<U> distribution);

  U update(final double observation,
           final U dist);


  //additive stats
//  Vec from(final U dist);
//
//  U to(final Vec stat);
//
//  Vec updateTo(final double observation, final Vec stat);
//
//  double likelihood(final Vec stat);

//  List<StatFunc> sufficientSpaceExtractors();

}


