package com.expleague.ml.methods.seq.automaton.evaluation;


import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.methods.seq.automaton.DFA;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.methods.seq.automaton.AutomatonStats;

import java.util.function.Function;

public class NonOptimizedStateEvaluation<T> implements Function<AutomatonStats<T>, Double> {
  @Override
  public Double apply(AutomatonStats<T> automatonStats) {
    final DataSet<Seq<T>> dataSet = automatonStats.getDataSet();
    final Vec target = automatonStats.getTarget();
    final DFA<T> automaton = automatonStats.getAutomaton();
    final double[] stateSum = new double[automaton.getStateCount()];
    final double[] stateSum2 = new double[automaton.getStateCount()];
    final int[] stateSize = new int[automaton.getStateCount()];

    for (int i = 0; i < target.length(); i++) {
      final int to = automaton.run(dataSet.at(i));
      final double w = target.get(i);
      stateSum[to] += w;
      stateSum2[to] += w * w;
      stateSize[to]++;
    }

    double score = 0;
    for (int i = 0; i < automaton.getStateCount(); i++) {
      if (stateSize[i] > 0)
        score += stateSum2[i] - stateSum[i] * stateSum[i] / stateSize[i];
    }

    return score;
  }
}

