package com.expleague.ml.methods.seq.automaton.evaluation;

import com.expleague.commons.math.MathTools;
import com.expleague.ml.methods.seq.automaton.AutomatonStats;

import java.util.function.Function;

public class OptimizedStateEvaluation<T> implements Function<AutomatonStats<T>, Double> {
  @Override
  public Double apply(AutomatonStats<T> automatonStats) {
    double score = 0;
    final int stateCount = automatonStats.getAutomaton().getStateCount();
    for (int i = 0; i < stateCount; i++) {
      final double sum = automatonStats.getStateSum().get(i);
      final double sum2 = automatonStats.getStateSum2().get(i);
      final double size = automatonStats.getStateWeight().get(i);
      if (size > MathTools.EPSILON) {
        score += sum2 - sum * sum / size;
      }
    }

    return score;
  }
}
