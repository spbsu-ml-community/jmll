package com.spbsu.exp.modelexp;

/**
 * User: solar
 * Date: 02.04.15
 * Time: 19:18
 */
public interface Experiment {
  double work(Query q);
  boolean relevant(Query q);
  double realScore();
}
