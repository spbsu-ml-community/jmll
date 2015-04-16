package com.spbsu.exp.modelexp;

/**
 * User: solar
 * Date: 02.04.15
 * Time: 19:18
 */
public interface User {
  void feedback(double score);
  double activity();
  Query next(int day, int hour);
}
