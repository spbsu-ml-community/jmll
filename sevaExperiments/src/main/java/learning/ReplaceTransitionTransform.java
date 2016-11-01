package learning;

public class ReplaceTransitionTransform implements Transform {
  private final int from;
  private final int to;
  private final int c;
  private Transform addTransform;
  private Transform removeTransform;

  ReplaceTransitionTransform(int from, int to, int c) {
    this.from = from;
    this.to = to;
    this.c = c;
  }
  @Override
  public void applyTransform(IncrementalAutomatonBuilder.LearningState learningState) {
    removeTransform = new RemoveTransitionTransform(from, c);
    addTransform = new AddTransitionTransform(from, to, c);
    removeTransform.applyTransform(learningState);
    addTransform.applyTransform(learningState);
  }

  @Override
  public void cancelTransform(IncrementalAutomatonBuilder.LearningState learningState) {
    addTransform.cancelTransform(learningState);
    removeTransform.cancelTransform(learningState);
  }

  @Override
  public String getDescription() {
    return String.format("Redirect edge from %d by %d to %d", from, c, to);
  }
}
