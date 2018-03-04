package com.expleague.ml.distributions;

import com.expleague.commons.math.vectors.Vec;

/**
 * Created by noxoomo on 11/02/2018.
 */
public class VecStatisticsAggregator implements NumericAggregator<VecSufficientStat> {
  final CombineFunction[] trans;

  public VecStatisticsAggregator(final CombineFunction[] trans) {
    this.trans = trans;
  }

  @Override
  public NumericAggregator<VecSufficientStat> update(double observation,
                                                     final double w,
                                                     VecSufficientStat stat) {
    final Vec dst = stat.stats;
    dst.adjust(0, w);

    for (int i = 0; i < trans.length; ++i) {
      final double val = trans[i].combine(dst.get(i + 1), observation, w);
      stat.stats.set(i + 1, val);
    }
    return this;
  }

  public VecSufficientStat create() {
    return new VecSufficientStat(trans.length + 1);
  }



  public static CombineFunction sumCombine() {
    return (stat, newObs, w) -> stat + w * newObs;
  }

  public static CombineFunction squareSumCombine() {
    return (stat, newObs, w) -> stat + w * newObs * newObs;
  }

  public static CombineFunction logSumCombine() {
    return (stat, newObs, w) -> stat + (w > 0 ? w * Math.log(newObs) : 0);
  }

  public static CombineFunction expSumCombine() {
    return (stat, newObs, w) -> stat + w * Math.exp(newObs);
  }

  public static CombineFunction maxCombine() {
    return (stat, newObs, w) -> Math.max(stat, newObs);
  }

  public static CombineFunction minCombine() {
    return (stat, newObs, w) -> Math.min(stat, newObs);
  }



  public interface CombineFunction {
    double combine(double stat, double newObs, double w);
  }


}

