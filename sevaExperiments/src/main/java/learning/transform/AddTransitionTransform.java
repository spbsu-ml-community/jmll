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
  public LearningState<T> applyTransform(LearningState<T> learningState) {
    final LearningState<T> result = new LearningState<>(learningState);
    final DFA<T> automaton = result.getAutomaton();
    final int stateCount = automaton.getStateCount();
    final List<Seq<T>> data = learningState.getData();
    final TIntList classes = learningState.getClasses();
    final TDoubleList weights = learningState.getWeights();
    final List<TIntIntMap> samplesEndState = learningState.getSamplesEndState();

    final List<TIntSet> newVia = new ArrayList<>(stateCount);
    final List<double[]> newStateClassWeight = new ArrayList<>(stateCount);
    final List<TIntIntMap> newSamplesEndState = new ArrayList<>(stateCount);

    learningState.getStateClassWeight().forEach(stateClassCount -> {
      newStateClassWeight.add(Arrays.copyOf(stateClassCount, learningState.getClassCount()));
      newVia.add(null);
      newSamplesEndState.add(null);
    });

    automaton.addTransition(from, to, c);
    newSamplesEndState.set(from, new TIntIntHashMap(samplesEndState.get(from)));
    samplesEndState.get(from).forEachEntry((index, endI) -> {
      int curState = from;
      int newEndI = endI;
      final Seq<T> word = data.get(index);
      for (int i = endI; i < word.length(); i++, newEndI = i) {
        final T c = word.at(i);
        if (!automaton.hasTransition(curState, c)) {
          break;
        }
        curState = automaton.getTransition(curState, c);
        if (newVia.get(curState) == null) {
          newVia.set(curState, new TIntHashSet(learningState.getSamplesViaState().get(curState)));
        }
        newVia.get(curState).add(index);
      }

      newStateClassWeight.get(from)[classes.get(index)] -= weights.get(index);
      newStateClassWeight.get(curState)[classes.get(index)] += weights.get(index);
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
    result.setStateClassWeight(newStateClassWeight);
    return result;
  }

  @Override
  public String getDescription() {
    return String.format("Add transition by %s from %d to %d", c.toString(), from, to);
  }
}
