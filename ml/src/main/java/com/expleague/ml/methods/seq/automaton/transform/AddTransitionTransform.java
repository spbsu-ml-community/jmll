package com.expleague.ml.methods.seq.automaton.transform;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.methods.seq.automaton.DFA;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.methods.seq.automaton.AutomatonStats;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.ArrayList;
import java.util.List;

public class AddTransitionTransform<T> implements Transform<T> {
  private final int from;
  private final int to;
  private final T c;

  public AddTransitionTransform(int from, int to, T c) {
    this.from = from;
    this.to = to;
    this.c = c;
  }

  @Override
  public AutomatonStats<T> applyTransform(AutomatonStats<T> automatonStats) {
    final AutomatonStats<T> result = new AutomatonStats<>(automatonStats);
    final DFA<T> automaton = result.getAutomaton();
    final int stateCount = automaton.getStateCount();
    final DataSet<Seq<T>> data = automatonStats.getDataSet();
    final Vec target = automatonStats.getTarget();
    final Vec weights = automatonStats.getWeights();
    final List<TIntIntMap> samplesEndState = automatonStats.getSamplesEndState();

    final List<TIntSet> newVia = new ArrayList<>(stateCount);
    final List<TIntIntMap> newSamplesEndState = new ArrayList<>(stateCount);

    final TDoubleList newStateWeight = new TDoubleArrayList(automatonStats.getStateWeight());
    final TDoubleList newStateSum = new TDoubleArrayList(automatonStats.getStateSum());
    final TDoubleList newStateSum2 = new TDoubleArrayList(automatonStats.getStateSum2());

    for (int i = 0; i < stateCount; i++) {
      newVia.add(null);
      newSamplesEndState.add(null);
    }

    automaton.addTransition(from, to, c);
    newSamplesEndState.set(from, new TIntIntHashMap(samplesEndState.get(from)));
    samplesEndState.get(from).forEachEntry((index, endI) -> {
      int curState = from;
      int newEndI = endI;
      final Seq<T> word = data.at(index);
      for (int i = endI; i < word.length(); i++, newEndI = i) {
        final T c = word.at(i);
        if (!automaton.hasTransition(curState, c)) {
          break;
        }
        curState = automaton.getTransition(curState, c);
        if (newVia.get(curState) == null) {
          newVia.set(curState, new TIntHashSet(automatonStats.getSamplesViaState().get(curState)));
        }
        newVia.get(curState).add(index);
      }


      final double w = target.get(index);
      newStateSum.set(from, newStateSum.get(from) - w);
      newStateSum2.set(from, newStateSum2.get(from) - w * w);
      newStateWeight.set(from, newStateWeight.get(from) - weights.get(index));

      newStateSum.set(curState, newStateSum.get(curState) + w);
      newStateSum2.set(curState, newStateSum2.get(curState) + w * w);
      newStateWeight.set(curState, newStateWeight.get(curState) + weights.get(index));

      if (newSamplesEndState.get(curState) == null) {
        newSamplesEndState.set(curState, new TIntIntHashMap(samplesEndState.get(curState)));
      }
      newSamplesEndState.get(from).remove(index);
      newSamplesEndState.get(curState).put(index, newEndI);
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

    result.setStateWeight(newStateWeight);
    result.setStateSum(newStateSum);
    result.setStateSum2(newStateSum2);

    return result;
  }

  @Override
  public String getDescription() {
    return String.format("Add transition by %s from %d to %d", c.toString(), from, to);
  }
}
