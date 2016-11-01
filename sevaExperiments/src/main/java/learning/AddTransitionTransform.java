package learning;

import automaton.DFA;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.TIntSet;

import java.util.ArrayList;
import java.util.List;

public class AddTransitionTransform implements Transform {
  private final int from;
  private final int to;
  private final int c;

  private List<TIntList> toRemoveVia;
  private TIntList oldFalseEndCount;
  private TIntList oldTrueEndCount;
  private List<TIntIntMap> erasedEndState;
  private List<TIntIntMap> insertedEndState;

  AddTransitionTransform(int from, int to, int c) {
    this.from = from;
    this.to = to;
    this.c = c;
  }

  @Override
  public void applyTransform(IncrementalAutomatonBuilder.LearningState learningState) {
    final DFA automaton = learningState.getAutomaton();
    final List<TIntList> data = learningState.getData();
    final List<Boolean> classes = learningState.getClasses();
    final TIntList trueStringEndCount = learningState.getTrueStringEndCount();
    final TIntList falseStringEndCount = learningState.getFalseStringEndCount();
    final List<TIntIntMap> samplesEndState = learningState.getSamplesEndState();

    toRemoveVia = new ArrayList<>();
    erasedEndState = new ArrayList<>();
    insertedEndState = new ArrayList<>();
    for (int i = 0; i < automaton.getStateCount(); i++) {
      toRemoveVia.add(new TIntArrayList());
      erasedEndState.add(new TIntIntHashMap());
      insertedEndState.add(new TIntIntHashMap());
    }
    oldFalseEndCount = new TIntArrayList(falseStringEndCount);
    oldTrueEndCount = new TIntArrayList(trueStringEndCount);

    automaton.addTransition(from, to, c);
    samplesEndState.get(from).forEachEntry((index, endI) -> {
      int curState = from;
      int newEndI = endI;
      final TIntList word = data.get(index);
      for (int i = endI; i < word.size(); i++, newEndI = i) {
        final int c = word.get(i);
        if (!automaton.hasTransition(curState, c)) {
          break;
        }
        curState = automaton.getTransition(curState, c);
        final TIntSet stringsViaCurState = learningState.getSamplesViaState().get(curState);
        if (!stringsViaCurState.contains(index)) {
          stringsViaCurState.add(index);
          toRemoveVia.get(curState).add(index);
        }
      }

      erasedEndState.get(from).put(index, endI);
      insertedEndState.get(curState).put(index, newEndI);

      if (classes.get(index)) {
        trueStringEndCount.set(from, trueStringEndCount.get(from) - 1);
        trueStringEndCount.set(curState, trueStringEndCount.get(curState) + 1);
      } else {
        falseStringEndCount.set(from, falseStringEndCount.get(from) - 1);
        falseStringEndCount.set(curState, falseStringEndCount.get(curState) + 1);
      }

      return true;
    });

    for (int state = 0; state < automaton.getStateCount(); state++) {
      final int state1 = state;
      erasedEndState.get(state).forEachKey(index -> {
        samplesEndState.get(state1).remove(index);
        return true;
      });
      insertedEndState.get(state).forEachEntry((index, endI) -> {
        samplesEndState.get(state1).put(index, endI);
        return true;
      });
    }
  }

  @Override
  public void cancelTransform(IncrementalAutomatonBuilder.LearningState learningState) {
    if (toRemoveVia == null) {
      throw new IllegalStateException("You should apply a transform before cancelling it");
    }

    learningState.getAutomaton().removeTransition(from, c);
    for (int i = 0; i < learningState.getAutomaton().getStateCount(); i++) {
      final int i1 = i;
      insertedEndState.get(i).forEachKey(index -> {
        learningState.getSamplesEndState().get(i1).remove(index);
        return true;
      });

      erasedEndState.get(i).forEachEntry((index, endI) -> {
        learningState.getSamplesEndState().get(i1).put(index, endI);
        return true;
      });

      toRemoveVia.get(i).forEach(index -> {
        learningState.getSamplesViaState().get(i1).remove(index);
        return true;
      });
    }
    learningState.setTrueStringEndCount(oldTrueEndCount);
    learningState.setFalseStringEndCount(oldFalseEndCount);

    toRemoveVia = null;
    erasedEndState = null;
    insertedEndState = null;
    oldFalseEndCount = null;
    oldTrueEndCount = null;
  }

  @Override
  public String getDescription() {
    return String.format("Add transition by %d from %d to %d", c, from, to);
  }
}
