package learning;

import automaton.DFA;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.hash.TIntHashSet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SplitStateTransform implements Transform {
  private final int state;
  private final int alphabetSize;
  private List<Transform> transforms;

  SplitStateTransform(int state, int alphabetSize) {
    this.state = state;
    this.alphabetSize = alphabetSize;
  }

  @Override
  public void applyTransform(IncrementalAutomatonBuilder.LearningState learningState) {
    final DFA automaton = learningState.getAutomaton();
    transforms = new ArrayList<>();
    for (int c = 0; c < alphabetSize; c++) {
      if (!automaton.hasTransition(state, c)) {
        learningState.getFalseStringEndCount().add(automaton.getStartState());
        learningState.getTrueStringEndCount().add(automaton.getStartState());
        learningState.getSamplesEndState().add(new TIntIntHashMap());
        learningState.getSamplesViaState().add(new TIntHashSet());
        final int newState = automaton.createNewState();
        transforms.add(new AddTransitionTransform(state, newState, c));
      }
    }

    transforms.forEach(transform -> transform.applyTransform(learningState));
  }

  @Override
  public void cancelTransform(IncrementalAutomatonBuilder.LearningState learningState) {
    if (transforms == null) {
      throw new IllegalStateException("You should apply a transform before cancelling it");
    }

    final DFA automaton = learningState.getAutomaton();
    Collections.reverse(transforms);
    transforms.forEach(transform -> {
      transform.cancelTransform(learningState);
      automaton.removeState(automaton.getStateCount() - 1);
      final int lastState = automaton.getStateCount();
      learningState.getFalseStringEndCount().remove(lastState);
      learningState.getTrueStringEndCount().remove(lastState);
      learningState.getSamplesEndState().remove(lastState);
      learningState.getSamplesViaState().remove(lastState);
    });
    transforms = null;
  }

  @Override
  public String getDescription() {
    return "Split state " + state;
  }
}
