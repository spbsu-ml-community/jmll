package com.spbsu.ml.methods.seq.automaton.transform;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.methods.seq.automaton.AutomatonStats;
import com.spbsu.ml.methods.seq.automaton.DFA;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.ArrayList;
import java.util.List;

public class RemoveTransitionTransform<T> implements Transform<T> {
  private final int from;
  private final T c;
  private int to;

  public RemoveTransitionTransform(int from, T c) {
    this.from = from;
    this.c = c;
  }

  @Override
  public AutomatonStats<T> applyTransform(AutomatonStats<T> automatonStats) {
    final AutomatonStats<T> result = new AutomatonStats<>(automatonStats);
    final DFA<T> automaton = result.getAutomaton();
    final DataSet<Seq<T>> data = automatonStats.getDataSet();
    final Vec target = automatonStats.getTarget();
    final Vec weights = automatonStats.getWeights();
    final List<TIntIntMap> samplesEndState = automatonStats.getSamplesEndState();
    final List<TIntSet> samplesViaState = automatonStats.getSamplesViaState();
    final int stateCount = automaton.getStateCount();

    final List<TIntSet> newVia = new ArrayList<>(stateCount);
    final List<TIntIntMap> newSamplesEndState = new ArrayList<>(stateCount);

    final TDoubleList newStateWeights = new TDoubleArrayList(automatonStats.getStateWeight());
    final TDoubleList newStateSum = new TDoubleArrayList(automatonStats.getStateSum());
    final TDoubleList newStateSum2 = new TDoubleArrayList(automatonStats.getStateSum2());

    for (int i = 0; i < stateCount; i++) {
      newVia.add(null);
      newSamplesEndState.add(null);
    }

    to = automaton.getTransition(from, c);
    automaton.removeTransition(from, c);

    final TIntSet statesVia = new TIntHashSet();
    samplesViaState.get(from).forEach(index -> {
      final Seq<T> word = data.at(index);
      int curState = automaton.getStartState();
      int endI = 0;

      statesVia.clear();
      statesVia.add(curState);
      for (int i = 0; i < word.length(); i++, endI = i) {
        final int to = automaton.getTransition(curState, word.at(i));
        if (to == -1) {
          break;
        }
        curState = to;
        statesVia.add(curState);
      }

      final double w = target.get(index);

      for (int i = 0; i < stateCount; i++) {
        if (samplesViaState.get(i).contains(index) && !statesVia.contains(i)) {
          if (newVia.get(i) == null) {
            newVia.set(i, new TIntHashSet(samplesViaState.get(i)));
          }
          newVia.get(i).remove(index);
        }
        if (samplesEndState.get(i).containsKey(index)) {
          if (newSamplesEndState.get(i) == null) {
            newSamplesEndState.set(i, new TIntIntHashMap(samplesEndState.get(i)));
          }
          newSamplesEndState.get(i).remove(index);
          newStateSum.set(i, newStateSum.get(i) - w);
          newStateSum2.set(i, newStateSum2.get(i) - w * w);
          newStateWeights.set(i, newStateWeights.get(i) - weights.get(index));
        }
      }

      if (newSamplesEndState.get(curState) == null) {
        newSamplesEndState.set(curState, new TIntIntHashMap(samplesEndState.get(curState)));
      }
      newSamplesEndState.get(curState).put(index, endI);
      newStateSum.set(curState, newStateSum.get(curState) + w);
      newStateSum2.set(curState, newStateSum2.get(curState) + w * w);
      newStateWeights.set(curState, newStateWeights.get(curState) + weights.get(index));

      return true;
    });

    for (int i = 0; i < stateCount; i++) {
      if (newVia.get(i) != null) {
        result.getSamplesViaState().set(i, newVia.get(i));
      }
      if (newSamplesEndState.get(i) != null) {
        result.getSamplesEndState().set(i, newSamplesEndState.get(i));
      }
    }
    result.setStateWeight(newStateWeights);
    result.setStateSum(newStateSum);
    result.setStateSum2(newStateSum2);

    return result;
  }

  @Override
  public String getDescription() {
    return String.format("Remove edge from %d to %d by %s", from, to, c.toString());
  }
}
