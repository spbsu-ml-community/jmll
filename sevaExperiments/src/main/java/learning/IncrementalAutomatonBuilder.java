package learning;

import automaton.DFA;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.commons.seq.regexp.Matcher;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TDoubleArrayList;
import learning.transform.*;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.function.Function;

public class IncrementalAutomatonBuilder<T> {
  private final static double INF = 1e18;
  private final static int MAX_STATE_COUNT = 15;
  private final Function<LearningState<T>, Double> costFunction;
  
  public IncrementalAutomatonBuilder(final Function<LearningState<T>, Double> costFunction) {
    this.costFunction = costFunction;  
  }

  public DFA<T> buildAutomaton(final List<Seq<T>> data, final TIntList classes,
                            final Alphabet<T> alphabet, final int classCount, final int iterCount) {
    TDoubleList weights = new TDoubleArrayList(data.size());
    for (int i = 0; i < data.size(); i++) {
      weights.add(1.0 / data.size());
    }
    return buildAutomaton(data, classes, weights, alphabet, classCount, iterCount);
  }

  public DFA<T> buildAutomaton(final List<Seq<T>> data, final TIntList classes, final TDoubleList weights,
                               final Alphabet<T> alphabet, final int classCount, final int iterCount) {
    if (data.size() != classes.size()) {
      throw new IllegalArgumentException("");
    }

    LearningState<T> learningState = new LearningState<>(alphabet, classCount, data, classes, weights);

    double oldCost = costFunction.apply(learningState);
    for (int iter = 0; iter < iterCount; iter++) {

      double optCost = INF;
      Transform<T> optTransform = null;
      for (Transform<T> transform: getTransforms(learningState)) {
        LearningState<T> newLearningState = transform.applyTransform(learningState);
        final double newCost = costFunction.apply(newLearningState);
        if (newCost < optCost) {
          optCost = newCost;
          optTransform = transform;
        }
      }
      if (optTransform == null || optCost >= oldCost - 1e-9) {
        break;
      }
      learningState = optTransform.applyTransform(learningState);
      removeUnreachableStates(learningState);
      System.out.printf("Iter=%d, transform=%s, newCost=%f, state count=%d\n",
              iter, optTransform.getDescription(), optCost, learningState.getAutomaton().getStateCount());
      oldCost = optCost;
    }
    return finalizeAutomaton(learningState);
  }

  private DFA<T> finalizeAutomaton(final LearningState<T> learningState) {
    final int classCount = learningState.getClassCount();
    final DFA<T> automaton = learningState.getAutomaton();
    final int stateCount = automaton.getStateCount();
    final Alphabet<T> alphabet = learningState.getAlphabet();

    final int[] endStates = new int[classCount];

    for (int i = 0; i < classCount; i++) {
      endStates[i] = automaton.createNewState(i);
    /*  for (int c = 0; c < learningState.alphabetSize + 1; c++) {
        automaton.addTransition(endStates[i], endStates[i], c);
      }*/
    }
    for (int i = 0; i < stateCount; i++) {
      int maxClass = 0;
      final double[] stateClassWeight = learningState.getStateClassWeight().get(i);
      for (int clazz = 1; clazz < classCount; clazz++) {
        if (stateClassWeight[clazz] > stateClassWeight[maxClass]) {
          maxClass = clazz;
        }
      }
      automaton.setStateClass(i, maxClass); // TODO remove it
      automaton.addTransition(i, endStates[maxClass], alphabet.getT(Matcher.Condition.ANY));
/*      for (int c = 0; c < learningState.alphabetSize; c++) {
        if (!automaton.hasTransition(i, c)) {
          automaton.addTransition(i, endStates[maxClass], c);
        }
      }*/
    }
    return automaton;
  }


  private List<Transform<T>> getTransforms(final LearningState<T> learningState) {
    final DFA<T> automaton = learningState.getAutomaton();
    final int stateCount = automaton.getStateCount();
    final Alphabet<T> alphabet = learningState.getAlphabet();
    final List<Transform<T>> transforms = new ArrayList<>();
    for (int from = 0; from < stateCount; from++) {
      for (int c = 0; c < alphabet.size(); c++) {
        if (automaton.hasTransition(from, alphabet.getT(alphabet.get(c)))) {
          transforms.add(new RemoveTransitionTransform<>(from, alphabet.getT(alphabet.get(c))));
          for (int to = 0; to < stateCount; to++) {
            if (to != from) {
              transforms.add(new ReplaceTransitionTransform<>(from, to, alphabet.getT(alphabet.get(c))));
            }
          }
        }
      }
      if (stateCount < MAX_STATE_COUNT) {
        for (int to = 0; to < stateCount; to++) {
//          transforms.add(new SplitStateTransform(from, alphabetSize));
          for (int c = 0; c < alphabet.size(); c++) {
            if (!automaton.hasTransition(from, alphabet.getT(alphabet.get(c)))) {
              T cT = alphabet.getT(alphabet.get(c));
              transforms.add(new AddTransitionTransform<>(from, to, cT));
              for (int c1 = 0; c1 < alphabet.size(); c1++) {
                transforms.add(new AddNewStateTransform<>(from, to, cT, alphabet.getT(alphabet.get(c1))));
              }
            }
          }
        }
      }
    }

    return transforms;
  }

  private void removeUnreachableStates(final LearningState<T> learningState) {
    final DFA<T> automaton = learningState.getAutomaton();
    final Queue<Integer> queue = new LinkedList<>();
    final Alphabet<T> alphabet = learningState.getAlphabet();
    queue.add(automaton.getStartState());
    final boolean[] reached = new boolean[automaton.getStateCount()];
    reached[automaton.getStartState()] = true;

    while (!queue.isEmpty()) {
      final int v = queue.poll();
      for (int c = 0; c < learningState.getAlphabet().size(); c++) {
        final int to = automaton.getTransition(v, alphabet.getT(alphabet.get(c)));
        if (to != -1 && !reached[to]) {
          queue.add(to);
          reached[to] = true;
        }
      }
    }
    for (int i = automaton.getStateCount() - 1; i >= 0; i--) {
      if (!reached[i] && i != automaton.getStartState()) {
        automaton.removeState(i);
        learningState.getSamplesEndState().remove(i);
        learningState.getStateClassWeight().remove(i);
        learningState.getSamplesViaState().remove(i);
      }
    }
  }
}
