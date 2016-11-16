package learning;

import java.util.List;
import java.util.function.Function;

public class OptimizedCostFunction<T> implements Function<LearningState<T>, Double> {
  private final static double LAMBDA = 0.1;
  private final static double STATE_SIZE_COST_THRESHOLD = 0.1;

  @Override
  public Double apply(LearningState<T> learningState) {
    double cost = 0;
    double stateSizeCost = 0;
    final int stateCount = learningState.getAutomaton().getStateCount();
    final List<double[]> classStateWeight = learningState.getStateClassWeight();
    for (int i = 0; i < stateCount; i++) {
      double maxClassWeight = 0;
      double stateWeight = 0;
      for (int clazz = 0; clazz < learningState.getClassCount(); clazz++) {
        stateWeight += classStateWeight.get(i)[clazz];
        maxClassWeight = Double.max(maxClassWeight, classStateWeight.get(i)[clazz]);
      }

      cost += stateWeight - maxClassWeight;
      if (1.0 * (stateWeight - maxClassWeight) / stateWeight > STATE_SIZE_COST_THRESHOLD) {
        stateSizeCost += 1.0 * stateWeight * stateWeight;
      }
    }

    return (cost + Math.sqrt(stateSizeCost) * LAMBDA);
  }
}
