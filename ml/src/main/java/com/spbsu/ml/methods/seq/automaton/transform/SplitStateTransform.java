package com.spbsu.ml.methods.seq.automaton.transform;

import com.spbsu.ml.methods.seq.automaton.AutomatonStats;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.hash.TIntHashSet;

public class SplitStateTransform<T> implements Transform<T> {
  private final int state;
  private final T c;

  public SplitStateTransform(int state, T c) {
    this.state = state;
    this.c = c;
  }

  @Override
  public AutomatonStats<T> applyTransform(AutomatonStats<T> automatonStats) {
    AutomatonStats<T> newAutomatonStats = new AutomatonStats<>(automatonStats);

    final int newState = newAutomatonStats.addNewState();
    return new AddTransitionTransform<>(state, newState, c).applyTransform(newAutomatonStats);
  }

  @Override
  public String getDescription() {
    return String.format("Split state %d, by %s", state, c.toString());
  }
}
