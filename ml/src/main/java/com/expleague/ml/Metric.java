package com.expleague.ml;

public interface Metric<T> {
  double distance(T a, T b);
}
