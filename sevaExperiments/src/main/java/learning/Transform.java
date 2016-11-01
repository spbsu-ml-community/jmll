package learning;

interface Transform {
  /**
   *
   * @param learningState state of learning
   */
  void applyTransform(IncrementalAutomatonBuilder.LearningState learningState);

  /**
   * Cancels previously applied transform if there were no other changes between.
   * Otherwise behaviour is undefined.
   * @param learningState state of learning
   */
  void cancelTransform(IncrementalAutomatonBuilder.LearningState learningState);
  String getDescription();
}
