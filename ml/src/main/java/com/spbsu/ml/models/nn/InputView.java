package com.spbsu.ml.models.nn;

public class InputView {
  public final int weightStart;
  public final int weightLength;
  public final int stateStart;
  public final int stateLength;

  public InputView(int weightStart, int weightLength, int stateStart, int stateLength) {
    this.weightStart = weightStart;
    this.weightLength = weightLength;
    this.stateStart = stateStart;
    this.stateLength = stateLength;
  }
}
