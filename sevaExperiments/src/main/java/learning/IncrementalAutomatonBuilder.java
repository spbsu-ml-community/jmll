package learning;

import automaton.DFA;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class IncrementalAutomatonBuilder {
  private final static double LAMBDA = 0.2;
  private final static double STATE_SIZE_COST_THRESHOLD = 0.1;
  private final static double INF = 1e18;
  private final static int MAX_STATE_COUNT = 10;

  public DFA buildAutomaton(List<TIntList> data, List<Boolean> classes, int alphabetSize, int iterCount) {
    if (data.size() != classes.size()) {
      throw new IllegalArgumentException("");
    }

    LearningState learningState = new LearningState(alphabetSize, data, classes);

    double oldCost = getFullCost(learningState);
    for (int iter = 0; iter < iterCount; iter++) {

      double optCost = INF;
      Transform optTransform = null;
      for (Transform transform: getTransforms(learningState, alphabetSize)) {
        transform.applyTransform(learningState);
        final double newCost = getFullCost(learningState);
        if (newCost < optCost) {
          optCost = newCost;
          optTransform = transform;
        }
        transform.cancelTransform(learningState);
      }
      if (optTransform == null || optCost >= oldCost) {
        break;
      }
      optTransform.applyTransform(learningState);
      removeUnreachableStates(learningState, alphabetSize);
      System.out.printf("Iter=%d, transform=%s, newCost=%f, accuracy=%f, state count=%d\n",
              iter, optTransform.getDescription(), optCost,
              1 - 1.0 * getFailedSamplesCount(learningState) / data.size(), learningState.automaton.getStateCount());
      oldCost = optCost;
    }
    return learningState.automaton;
  }

  private int getFailedSamplesCount(LearningState learningState) {
    int count = 0;
    final int stateCount = learningState.automaton.getStateCount();
    for (int i = 0; i < stateCount; i++) {
      if (learningState.automaton.isStateFinal(i)) {
        count += learningState.falseStringEndCount.get(i);
      } else {
        count += learningState.trueStringEndCount.get(i);
      }
    }
    return count;
  }

  private double getStateSizeCost(LearningState learningState) {
    double cost = 0;
    final int stateCount = learningState.automaton.getStateCount();
    for (int state = 0; state < stateCount; state++) {
      final int falseEnds = learningState.falseStringEndCount.get(state);
      final int trueEnds = learningState.trueStringEndCount.get(state);
      final int stateSize = falseEnds + trueEnds;
      if (1.0 * Integer.min(falseEnds, trueEnds) / stateSize > STATE_SIZE_COST_THRESHOLD) {
        cost += 1.0 * stateSize * stateSize;
      }
    }
    return Math.sqrt(cost) * LAMBDA / learningState.data.size();
  }

  private double getFullCost(LearningState learningState) {
    final List<TIntList> data = learningState.data;
    return 1.0 * getFailedSamplesCount(learningState) / data.size() + getStateSizeCost(learningState);
  }

  List<Transform> getTransforms(LearningState learningState, int alphabetSize) {
    final int stateCount = learningState.automaton.getStateCount();
    final DFA automaton = learningState.automaton;
    final List<Transform> transforms = new ArrayList<>();
    for (int from = 0; from < stateCount; from++) {
      for (int c = 0; c < alphabetSize; c++) {
        if (automaton.hasTransition(from, c)) {
          transforms.add(new RemoveTransitionTransform(from, c));
          for (int to = 0; to < stateCount; to++) {
            if (to != from) {
              transforms.add(new ReplaceTransitionTransform(from, to, c));
            }
          }
        }
      }
      if (stateCount < MAX_STATE_COUNT) {
        for (int to = 0; to < stateCount; to++) {
          transforms.add(new SplitStateTransform(from, alphabetSize));
          for (int c = 0; c < alphabetSize; c++) {
            if (!automaton.hasTransition(from, c)) {
              transforms.add(new AddTransitionTransform(from, to, c));
              for (int c1 = 0; c1 < alphabetSize; c1++) {
                transforms.add(new AddNewStateTransform(from, to, c, c1));
              }
            }
          }
        }
      }
    }

    return transforms;
  }

  private void removeUnreachableStates(LearningState learningState, int alphabetSize) {
    final DFA automaton = learningState.automaton;
    final Queue<Integer> queue = new LinkedList<>();
    queue.add(automaton.getStartState());
    final boolean[] reached = new boolean[automaton.getStateCount()];
    reached[automaton.getStartState()] = true;

    while (!queue.isEmpty()) {
      final int v = queue.poll();
      for (int c = 0; c < alphabetSize; c++) {
        final int to = automaton.getTransition(v, c);
        if (to != -1 && !reached[to]) {
          queue.add(to);
          reached[to] = true;
        }
      }
    }
    for (int i = automaton.getStateCount() - 1; i >= 0; i--) {
      if (!reached[i] && i != automaton.getStartState() && i != learningState.finalState) {
        automaton.removeState(i);
      }
    }
  }

  static class LearningState {
    private final DFA automaton;
    private final int finalState;
    private List<TIntSet> samplesViaState = new ArrayList<>();
    private List<TIntIntMap> samplesEndState = new ArrayList<>();
    private TIntList falseStringEndCount = new TIntArrayList();
    private TIntList trueStringEndCount = new TIntArrayList();
    private List<TIntList> data;
    private List<Boolean> classes;

    LearningState(int alphabetSize, List<TIntList> data, List<Boolean> classes) {
      automaton = new DFA(alphabetSize);
      finalState = automaton.createNewState();
      automaton.markFinalState(finalState, true);
      this.data = data;
      this.classes = classes;
      final TIntSet allIndicesSet = new TIntHashSet();
      final TIntIntMap allIndicesMap = new TIntIntHashMap();
      int falseCount = 0;
      int trueCount = 0;

      for (int i = 0; i < data.size(); i++) {
        allIndicesSet.add(i);
        allIndicesMap.put(i, 0);
        if (classes.get(i)) {
          trueCount++;
        } else {
          falseCount++;
        }
      }

      samplesEndState.add(allIndicesMap);
      samplesViaState.add(allIndicesSet);
      falseStringEndCount.add(falseCount);
      trueStringEndCount.add(trueCount);

      samplesEndState.add(new TIntIntHashMap());
      samplesViaState.add(new TIntHashSet());
      falseStringEndCount.add(0);
      trueStringEndCount.add(0);
    }

    public DFA getAutomaton() {
      return automaton;
    }

    public int getFinalState() {
      return finalState;
    }

    public List<TIntSet> getSamplesViaState() {
      return samplesViaState;
    }

    public List<TIntIntMap> getSamplesEndState() {
      return samplesEndState;
    }

    public TIntList getFalseStringEndCount() {
      return falseStringEndCount;
    }

    public TIntList getTrueStringEndCount() {
      return trueStringEndCount;
    }

    public List<TIntList> getData() {
      return data;
    }

    public List<Boolean> getClasses() {
      return classes;
    }

    public void setSamplesViaState(List<TIntSet> samplesViaState) {
      this.samplesViaState = samplesViaState;
    }

    public void setSamplesEndState(List<TIntIntMap> samplesEndState) {
      this.samplesEndState = samplesEndState;
    }

    public void setFalseStringEndCount(TIntList falseStringEndCount) {
      this.falseStringEndCount = falseStringEndCount;
    }

    public void setTrueStringEndCount(TIntList trueStringEndCount) {
      this.trueStringEndCount = trueStringEndCount;
    }
  }
}
