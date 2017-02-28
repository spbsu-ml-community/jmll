package com.spbsu.ml.methods.seq.automaton;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.SeqOptimization;
import com.spbsu.ml.methods.seq.automaton.transform.AddNewStateTransform;
import com.spbsu.ml.methods.seq.automaton.transform.AddTransitionTransform;
import com.spbsu.ml.methods.seq.automaton.transform.SplitStateTransform;
import com.spbsu.ml.methods.seq.automaton.transform.Transform;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class IncrementalAutomatonBuilder<T, Loss extends L2> implements SeqOptimization<T, Loss> {
  private final int maxStateCount;
  private final Alphabet<T> alphabet;
  private final Computable<AutomatonStats<T>, Double> stateEvaluation;
  private final int maxIterations;

  public IncrementalAutomatonBuilder(final Alphabet<T> alphabet,
                                     final Computable<AutomatonStats<T>, Double> stateEvaluation,
                                     final int maxStateCount,
                                     final int maxIterations) {
    this.alphabet = alphabet;
    this.stateEvaluation = stateEvaluation;
    this.maxStateCount = maxStateCount;
    this.maxIterations = maxIterations;
  }

  @Override
  public Computable<Seq<T>, Vec> fit(final DataSet<Seq<T>> learn, final Loss loss) {
    AutomatonStats<T> automatonStats = new AutomatonStats<>(alphabet, learn, loss.target());

    double oldCost = stateEvaluation.compute(automatonStats);

    for (int iter = 0; iter < maxIterations; iter++) {

      double optCost = Double.MAX_VALUE;
      Transform<T> optTransform = null;
      for (Transform<T> transform: getTransforms(automatonStats)) {
        final AutomatonStats<T> newAutomatonStats = transform.applyTransform(automatonStats);
        final double newCost = stateEvaluation.compute(newAutomatonStats);
        if (newCost < optCost) {
          optCost = newCost;
          optTransform = transform;
        }
      }
      if (optTransform == null || optCost >= oldCost - 1e-9) {
        break;
      }
      automatonStats = optTransform.applyTransform(automatonStats);
      removeUnreachableStates(automatonStats);
      //System.out.printf("Iter=%d, transform=%s, newCost=%f, state count=%d\n",
      //      iter, optTransform.getDescription(), optCost, automatonStats.getAutomaton().getStateCount());
      oldCost = optCost;
    }

    final DFA<T> automaton = automatonStats.getAutomaton();
    final double[] stateValue = new double[automaton.getStateCount()];
    for (int i = 0; i < automaton.getStateCount(); i++) {
      stateValue[i] = automatonStats.getStateSum().get(i) / automatonStats.getStateSize().get(i);
    }

    return argument -> new SingleValueVec(stateValue[automaton.run(argument)]);
  }


  private List<Transform<T>> getTransforms(final AutomatonStats<T> automatonStats) {
    final DFA<T> automaton = automatonStats.getAutomaton();
    final int stateCount = automaton.getStateCount();
    final Alphabet<T> alphabet = automatonStats.getAlphabet();
    final List<Transform<T>> transforms = new ArrayList<>();

    for (int from = 0; from < stateCount; from++) {
      for (int c = 0; c < alphabet.size(); c++) {
        if (automaton.hasTransition(from, alphabet.getT(alphabet.get(c)))) {
          // todo commented out to improve performance
          // transforms.add(new RemoveTransitionTransform<>(from, alphabet.getT(alphabet.get(c))));
          for (int to = 0; to < stateCount; to++) {
            if (to != from) {
              // todo commented out to improve performance
              // transforms.add(new ReplaceTransitionTransform<>(from, to, alphabet.getT(alphabet.get(c))));
            }
          }
        } else {
          final T cT = alphabet.getT(alphabet.get(c));
          if (stateCount < maxStateCount) {
            transforms.add(new SplitStateTransform<>(from, cT));
          }
          for (int to = 0; to < stateCount; to++) {
            transforms.add(new AddTransitionTransform<>(from, to, cT));
          }
        }
      }
      if (stateCount < maxStateCount) {
        for (int to = 0; to < stateCount; to++) {
          for (int c = 0; c < alphabet.size(); c++) {
            final T cT = alphabet.getT(alphabet.get(c));
            if (!automaton.hasTransition(from, alphabet.getT(alphabet.get(c)))) {
              if (!automaton.hasTransition(from, alphabet.getT(alphabet.get(c))))
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

  private void removeUnreachableStates(final AutomatonStats<T> automatonStats) {
    final DFA<T> automaton = automatonStats.getAutomaton();
    final Queue<Integer> queue = new LinkedList<>();
    final Alphabet<T> alphabet = automatonStats.getAlphabet();
    queue.add(automaton.getStartState());
    final boolean[] reached = new boolean[automaton.getStateCount()];
    reached[automaton.getStartState()] = true;

    while (!queue.isEmpty()) {
      final int v = queue.poll();
      for (int c = 0; c < automatonStats.getAlphabet().size(); c++) {
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
        automatonStats.getSamplesEndState().remove(i);
        automatonStats.getStateSize().remove(i);
        automatonStats.getStateSum().remove(i);
        automatonStats.getStateSum2().remove(i);
        automatonStats.getSamplesViaState().remove(i);
      }
    }
  }
}
