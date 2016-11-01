package learning;

import automaton.DFA;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.hash.TIntHashSet;

public class AddNewStateTransform implements Transform {
  private final int from;
  private final int to;
  private final int c1;
  private final int c2;
  private int newState = -1;
  private Transform addFromTransition = null;
  private Transform addToTransition = null;

  AddNewStateTransform(int from, int to, int c1, int c2) {
    this.from = from;
    this.to = to;
    this.c1 = c1;
    this.c2 = c2;
  }

  @Override
  public void applyTransform(IncrementalAutomatonBuilder.LearningState learningState) {
    newState = learningState.getAutomaton().createNewState();
    learningState.getTrueStringEndCount().add(0);
    learningState.getFalseStringEndCount().add(0);
    learningState.getSamplesViaState().add(new TIntHashSet());
    learningState.getSamplesEndState().add(new TIntIntHashMap());

    addToTransition = new AddTransitionTransform(newState, to, c2);
    addFromTransition = new AddTransitionTransform(from, newState, c1);
    addToTransition.applyTransform(learningState);
    addFromTransition.applyTransform(learningState);
  }

  @Override
  public void cancelTransform(IncrementalAutomatonBuilder.LearningState learningState) {
    if (addFromTransition == null) {
      throw new IllegalStateException("You should apply a transform before cancelling it");
    }
    final DFA automaton = learningState.getAutomaton();

    addFromTransition.cancelTransform(learningState);
    addToTransition.cancelTransform(learningState);
    automaton.removeState(automaton.getStateCount() - 1);
    final int lastState = automaton.getStateCount();

    learningState.getTrueStringEndCount().remove(lastState);
    learningState.getFalseStringEndCount().remove(lastState);
    learningState.getSamplesViaState().remove(lastState);
    learningState.getSamplesEndState().remove(lastState);
  }

  @Override
  public String getDescription() {
    return String.format("Add new state %d, edge from %d by %d and edge to %d by %d", newState, from, c1, to, c2);
  }
}
