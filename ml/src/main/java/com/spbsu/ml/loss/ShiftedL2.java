package com.spbsu.ml.loss;

import org.jetbrains.annotations.NotNull;


import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.data.set.DataSet;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * Created by irlab on 27.03.2015.
 */
public class ShiftedL2 extends L2 {
  private Vec step1Scores;

  public ShiftedL2(final Vec target, final DataSet<?> owner) {
    this(target, owner, new ArrayVec(target.dim()));
  }

  public ShiftedL2(final Vec target, final DataSet<?> owner, final Vec step1Scores) {
    super(target, owner);
    this.step1Scores = step1Scores;
  }

  public void setStep1Scores(final Vec step1Scores) {
    this.step1Scores = step1Scores;
  }

  @NotNull
  @Override
  public Vec gradient(final Vec x) {
    // 2 * (step1[i] + x[i] - target[i])
    final Vec result = copy(x);
    append(result, step1Scores);
    scale(result, -1);
    append(result, target);
    scale(result, -2);
    return result;
  }

  @Override
  public double value(final Vec point) {
    final Vec x = copy(point);
    append(x, step1Scores);
    scale(x, -1);
    append(x, target);
    return Math.sqrt(sum2(x) / x.dim());
  }
}


