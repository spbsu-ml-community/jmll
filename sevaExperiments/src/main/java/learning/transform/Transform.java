package learning.transform;

import learning.LearningState;

public interface Transform<T> {
  /**
   *
   * @param learningState state of learning
   * @return Learning state after applying of transform.
   * Only changed arrays are copied TODO
   */
  LearningState<T> applyTransform(LearningState<T> learningState);

  String getDescription();
}
