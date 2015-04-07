package com.spbsu.ml.cli.gridsearch;

import java.lang.reflect.Array;

/**
* User: qdeee
* Date: 24.03.15
*/
public class ParametersGridEnumerator<T> {
  private final T[][] ranges;
  private final int[] currents;

  public ParametersGridEnumerator(final T[][] ranges) {
    this.ranges = ranges;
    this.currents = new int[ranges.length];
    this.currents[currents.length - 1] = -1;
  }

  public boolean advance() {
    int rangePos = currents.length - 1;
    while (rangePos >= 0 && currents[rangePos] == ranges[rangePos].length - 1) {
      rangePos--;
    }

    if (rangePos != -1) {
      currents[rangePos]++;
      for (int pos = rangePos + 1; pos < currents.length; pos++) {
        currents[pos] = 0;
      }
      return true;
    } else {
      return false;
    }
  }

  public T[] getParameters() {
    //noinspection unchecked
    final T[] parameters = (T[]) Array.newInstance(ranges.getClass().getComponentType().getComponentType(), currents.length);
    for (int i = 0; i < parameters.length; i++) {
      parameters[i] = ranges[i][currents[i]];
    }
    return parameters;
  }
}
