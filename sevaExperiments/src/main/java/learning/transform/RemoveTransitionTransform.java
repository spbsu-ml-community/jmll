package learning.transform;

import automaton.DFA;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import learning.LearningState;

import java.util.ArrayList;
import java.util.Arrays;
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
  public LearningState<T> applyTransform(LearningState<T> learningState) {
    final LearningState<T> result = new LearningState<>(learningState);
    final DFA<T> automaton = result.getAutomaton();
    final List<Seq<T>> data = learningState.getData();
    final TIntList classes = learningState.getClasses();
    final TDoubleList weights = learningState.getWeights();
    final List<TIntIntMap> samplesEndState = learningState.getSamplesEndState();
    final List<TIntSet> samplesViaState = learningState.getSamplesViaState();
    final int stateCount = automaton.getStateCount();

    final List<TIntSet> newVia = new ArrayList<>(stateCount);
    final List<double[]> newStateClassWeight = new ArrayList<>(stateCount);
    final List<TIntIntMap> newSamplesEndState = new ArrayList<>(stateCount);

    learningState.getStateClassWeight().forEach(stateClassCount -> {
      newStateClassWeight.add(Arrays.copyOf(stateClassCount, learningState.getClassCount()));
      newVia.add(null);
      newSamplesEndState.add(null);
    });

    to = automaton.getTransition(from, c);
    automaton.removeTransition(from, c);

    //newVia.set(from, new TIntHashSet());

    TIntSet statesVia = new TIntHashSet();
    samplesViaState.get(from).forEach(index -> {
      final Seq<T> word = data.get(index);
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

      final int clazz = classes.get(index);
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
          newStateClassWeight.get(i)[clazz] -= weights.get(index);
        }
      }

      if (newSamplesEndState.get(curState) == null) {
        newSamplesEndState.set(curState, new TIntIntHashMap(samplesEndState.get(curState)));
      }
      newSamplesEndState.get(curState).put(index, endI);
      newStateClassWeight.get(curState)[clazz] += weights.get(index);

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
    result.setStateClassWeight(newStateClassWeight);
    return result;
  }

  @Override
  public String getDescription() {
    return String.format("Remove edge from %d to %d by %s", from, to, c.toString());
  }
}
