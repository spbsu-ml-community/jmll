package com.spbsu.exp.modelexp;

import com.spbsu.commons.func.Evaluator;

/**
 * User: solar
 * Date: 02.04.15
 * Time: 19:42
 */
public interface Setup {
  void add(Experiment exp);
  void cancel(Experiment exp);

  Experiment[] assign(User u, Query q);
  Stat score(Experiment exp);

  void feedback(User user, Query query, Experiment[] config, double score);

  void nextDay();
}
