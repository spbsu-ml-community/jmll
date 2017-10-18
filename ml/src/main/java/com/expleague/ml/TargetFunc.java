package com.expleague.ml;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.set.DataSet;

import java.io.PrintStream;

/**
 * User: solar
 * Date: 15.07.14
 * Time: 10:37
 */
public interface TargetFunc extends Func {
  DataSet<?> owner();

  default void printResult(Vec x, PrintStream out) {
    out.append(getClass().getName()).append(": ")
        .append(String.valueOf(value(x))).append("\n");
  }
}
