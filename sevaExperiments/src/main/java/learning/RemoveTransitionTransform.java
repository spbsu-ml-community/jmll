package learning;

import automaton.DFA;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.ArrayList;
import java.util.List;

public class RemoveTransitionTransform implements Transform {
  private final int from;
  private final int c;
  private int to;

  private List<TIntList> toAddVia;
  private TIntList oldFalseEndCount;
  private TIntList oldTrueEndCount;
  private List<TIntIntMap> erasedEndState;
  private List<TIntIntMap> insertedEndState;

  RemoveTransitionTransform(int from, int c) {
    this.from = from;
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
    final List<TIntSet> samplesViaState = learningState.getSamplesViaState();

    toAddVia = new ArrayList<>();
    erasedEndState = new ArrayList<>();
    insertedEndState = new ArrayList<>();
    for (int i = 0; i < automaton.getStateCount(); i++) {
      toAddVia.add(new TIntArrayList());
      erasedEndState.add(new TIntIntHashMap());
      insertedEndState.add(new TIntIntHashMap());
    }
    oldFalseEndCount = new TIntArrayList(falseStringEndCount);
    oldTrueEndCount = new TIntArrayList(trueStringEndCount);

    to = automaton.getTransition(from, c);
    automaton.removeTransition(from, c);

    final int stateCount = automaton.getStateCount();
    final TIntSet statesVia = new TIntHashSet();

    samplesViaState.get(from).forEach(index -> {
      TIntList word = data.get(index);
      int curState = automaton.getStartState();
      int endI = 0;
      statesVia.clear();
      statesVia.add(curState);

      for (int i = 0; i < word.size(); i++, endI = i) {
        final int to = automaton.getTransition(curState, word.get(i));
        if (to == -1) {
          break;
        }
        curState = to;
        statesVia.add(curState);
      }

      for (int i = 0; i < stateCount; i++) {
        if (samplesViaState.get(i).contains(index) && !statesVia.contains(i)) {
          toAddVia.get(i).add(index);
        }
        if (samplesEndState.get(i).containsKey(index)) {
          erasedEndState.get(i).put(index, samplesEndState.get(i).get(index));
          samplesEndState.get(i).remove(index);
          if (classes.get(index)) {
            trueStringEndCount.set(i, trueStringEndCount.get(i) - 1);
          } else {
            falseStringEndCount.set(i, falseStringEndCount.get(i) - 1);
          }
        }
      }

      insertedEndState.get(curState).put(index, endI);
      samplesEndState.get(curState).put(index, endI);

      if (classes.get(index)) {
        trueStringEndCount.set(curState, trueStringEndCount.get(curState) + 1);
      } else {
        falseStringEndCount.set(curState, falseStringEndCount.get(curState) + 1);
      }

      return true;
    });

    for (int i = 0; i < stateCount; i++) {
      final int i1 = i;
      toAddVia.get(i).forEach(index -> samplesViaState.get(i1).remove(index));
    }
  }

  @Override
  public void cancelTransform(IncrementalAutomatonBuilder.LearningState learningState) {
    if (toAddVia == null) {
        throw new IllegalStateException("You should apply a transform before cancelling it");
    }

    DFA automaton = learningState.getAutomaton();
    automaton.addTransition(from, to, c);
    for (int i = 0; i < automaton.getStateCount(); i++) {
      final int i1 = i;
      insertedEndState.get(i).forEachKey(index -> {
        learningState.getSamplesEndState().get(i1).remove(index);
        return true;
      });

      erasedEndState.get(i).forEachEntry((index, endI) -> {
        learningState.getSamplesEndState().get(i1).put(index, endI);
        return true;
      });

      toAddVia.get(i).forEach(index -> {
        learningState.getSamplesViaState().get(i1).add(index);
        return true;
      });
    }
    learningState.setTrueStringEndCount(oldTrueEndCount);
    learningState.setFalseStringEndCount(oldFalseEndCount);

    toAddVia = null;
    erasedEndState = null;
    insertedEndState = null;
    oldFalseEndCount = null;
    oldFalseEndCount = null;
  }

  @Override
  public String getDescription() {
    return String.format("Remove edge from %d to %d by %d", from, to, c);
  }
}
