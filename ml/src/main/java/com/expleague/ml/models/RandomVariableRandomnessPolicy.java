package com.expleague.ml.models;

import com.expleague.ml.randomnessAware.ProcessRandomnessPolicy;

public enum RandomVariableRandomnessPolicy implements ProcessRandomnessPolicy {
  Sample,
  Expectation
}
