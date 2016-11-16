package learning;

import automaton.DFA;

import java.util.List;
import java.util.function.Function;

public class NonOptimizedCostFunction<T> implements Function<LearningState<T>, Double> {
  private final static double LAMBDA = 0.1;
  private final static double STATE_SIZE_COST_THRESHOLD = 0.1;

  @Override
  public Double apply(LearningState<T> learningState) {
    double cost = 0;
    double stateSizeCost = 0;
    final int stateCount = learningState.getAutomaton().getStateCount();
    final double[] stateWeight = new double[stateCount];
    final int classCount = learningState.getClassCount();
    final double[][] stateClassWeight = new double[stateCount][classCount];
    final DFA<T> automaton = learningState.getAutomaton();
    for (int i = 0; i < learningState.getData().size(); i++) {
      final int state = automaton.run(learningState.getData().get(i));
      final double w = learningState.getWeights().get(i);
      stateWeight[state] += w;
      stateClassWeight[state][learningState.getClasses().get(i)] += w;
    }
    for (int i = 0; i < stateCount; i++) {
      double maxClassWeight = 0;
      for (int clazz = 0; clazz < classCount; clazz++) {
        maxClassWeight = Double.max(maxClassWeight, stateClassWeight[i][clazz]);
      }

      cost += stateWeight[i] - maxClassWeight;
      if (1.0 * (stateWeight[i] - maxClassWeight) / stateWeight[i] > STATE_SIZE_COST_THRESHOLD) {
        stateSizeCost += 1.0 * stateWeight[i] * stateWeight[i];
      }
    }

    return (cost + Math.sqrt(stateSizeCost) * LAMBDA);
  }
}

