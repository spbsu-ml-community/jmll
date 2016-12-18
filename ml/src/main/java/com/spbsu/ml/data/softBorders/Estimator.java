package com.spbsu.ml.data.softBorders;

/**
 * Created by noxoomo on 06/11/2016.
 */
public interface Estimator<Sample> {
  Estimator<Sample> add(Sample sample);
}
