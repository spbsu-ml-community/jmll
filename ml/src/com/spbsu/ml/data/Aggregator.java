package com.spbsu.ml.data;

/**
 * User: solar
 * Date: 26.08.13
 * Time: 22:09
 */
public interface Aggregator {
  void append(int feature, byte bin, int origDSIndex);
}
