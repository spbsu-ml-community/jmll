package com.spbsu.bernulli;
import java.util.function.IntFunction;

/**
 * Created by noxoomo on 08/04/15.
 */

public class AicEM<Result extends FittedModel> {
  IntFunction<Result> factory;
  final int min;
  final int max;

  public AicEM(IntFunction<Result> factory) {
    this(factory,2, Integer.MAX_VALUE);
  }
  public AicEM(IntFunction<Result>  factory, int min, int max) {
    this.min = min;
    this.max = max;
    this.factory = factory;
  }

  public final Result fit() {
    double score = Double.POSITIVE_INFINITY;
    int k = min;
    Result result = null;
    while (k < max) {
      Result model =factory.apply(k);
      double currentScore = 2 *  model.complexity - 2*model.likelihood;
      if (currentScore > score)
        break;
      score = currentScore;
      result = model;
      ++k;
    }
    return result;
  }
}
