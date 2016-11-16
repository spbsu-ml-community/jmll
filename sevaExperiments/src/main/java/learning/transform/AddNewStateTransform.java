package learning.transform;

import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.hash.TIntHashSet;
import learning.LearningState;

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
  public LearningState<T> applyTransform(LearningState<T> learningState) {
    final LearningState<T> newLearningState = new LearningState<>(learningState);
    newState = newLearningState.getAutomaton().createNewState();
    newLearningState.getStateClassWeight().add(new double[learningState.getClassCount()]);
    newLearningState.getSamplesViaState().add(new TIntHashSet());
    newLearningState.getSamplesEndState().add(new TIntIntHashMap());

    LearningState<T> state1 = new AddTransitionTransform<>(newState, to, c2).applyTransform(newLearningState);
    return new AddTransitionTransform<>(from, newState, c1).applyTransform(state1);
  }

  @Override
  public String getDescription() {
    return String.format("Add new state %d, edge from %d by %s and edge to %d by %s",
            newState, from, c1.toString(), to, c2.toString());
  }
}
