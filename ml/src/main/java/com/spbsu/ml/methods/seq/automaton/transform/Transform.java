package com.spbsu.ml.methods.seq.automaton.transform;

import com.spbsu.ml.methods.seq.automaton.AutomatonStats;

public interface Transform<T> {
  /**
   *
   * @param automatonStats state of learning
   * @return Learning state after applying of transform.
   * Only changed arrays are copied TODO
   */
  AutomatonStats<T> applyTransform(AutomatonStats<T> automatonStats);

  String getDescription();
}
