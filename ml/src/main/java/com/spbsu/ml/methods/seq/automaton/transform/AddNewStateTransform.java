package com.spbsu.ml.methods.seq.automaton.transform;

import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.ml.methods.seq.automaton.AutomatonStats;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.hash.TIntHashSet;

public class AddNewStateTransform<T> implements Transform<T> {
  private final int from;
  private final int to;
  private final T c1;
  private final T c2;
  private int newState;

  public AddNewStateTransform(int from, int to, T c1, T c2) {
    this.from = from;
    this.to = to;
    this.c1 = c1;
    this.c2 = c2;
  }

  @Override
  public AutomatonStats<T> applyTransform(AutomatonStats<T> automatonStats) {
    AutomatonStats<T> newAutomatonStats = new AutomatonStats<>(automatonStats);

    newState = newAutomatonStats.getAutomaton().createNewState();

    newAutomatonStats.getSamplesViaState().add(new TIntHashSet());
    newAutomatonStats.getSamplesEndState().add(new TIntIntHashMap());
    newAutomatonStats.getStateSize().add(0);
    newAutomatonStats.getStateSum().add(0);
    newAutomatonStats.getStateSum2().add(0);

    Alphabet<T> alphabet = newAutomatonStats.getAlphabet();
    for (int i = 0; i < alphabet.size(); i++) {
      if (alphabet.index(c2) != i) {
        newAutomatonStats = new AddTransitionTransform<>(newState, newState, alphabet.getT(alphabet.get(i))).applyTransform(newAutomatonStats);
      }
    }

    AutomatonStats<T> state1 = new AddTransitionTransform<>(newState, to, c2).applyTransform(newAutomatonStats);
    if (state1.getAutomaton().hasTransition(from, c1)) {
      state1 = new RemoveTransitionTransform<>(from, c1).applyTransform(state1);
    }
    return new AddTransitionTransform<>(from, newState, c1).applyTransform(state1);
  }

  @Override
  public String getDescription() {
    return String.format("Add new state %d, edge from %d by %s and edge to %d by %s",
            newState, from, c1.toString(), to, c2.toString());
  }
}
