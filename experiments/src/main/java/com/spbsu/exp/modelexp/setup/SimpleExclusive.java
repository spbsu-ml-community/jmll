package com.spbsu.exp.modelexp.setup;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.MultiMap;
import com.spbsu.exp.modelexp.*;

import java.util.*;

/**
* User: solar
* Date: 02.04.15
* Time: 21:13
*/
public class SimpleExclusive implements Setup {
  private double slot = 0.05;
  private int expLen = 14;
  private double prodShare = 0.5;
  private final FastRandom rng;
  private Experiment prod = new Experiment() {
    @Override
    public double work(Query q) {
      return 0;
    }

    @Override
    public boolean relevant(Query q) {
      return true;
    }

    @Override
    public double realScore() {
      return 0;
    }
  };
  private final Set<Experiment> current = new HashSet<>();
  private final MultiMap<Integer, Experiment> deadlines = new MultiMap<>();
  private final List<Experiment> queue = new ArrayList<>();
  private final Map<Experiment, Stat> value = new HashMap<>();
  int index;

  public SimpleExclusive(FastRandom rng) {
    this.rng = rng;
    index = 0;
  }

  @Override
  public void add(Experiment exp) {
    queue.add(exp);
  }

  @Override
  public void cancel(Experiment exp) {
    current.remove(exp);
  }

  @Override
  public void nextDay() {
    { // experimental setup for the next day
      current.removeAll(deadlines.get(index));

      while (current.size() < (1 - prodShare) / slot && !queue.isEmpty()) {
        final Experiment next = queue.remove(0);
        current.add(next);
        deadlines.put(index + expLen, next);
      }
    }
    index++;

    System.out.println("Experiments: " + Arrays.toString(current.toArray(new Experiment[current.size()])));
  }

  @Override
  public Experiment[] assign(User u, Query q) {
    final double v = rng.nextDouble();
    double sum = prodShare;
    final Iterator<Experiment> it = current.iterator();
    Experiment result = prod;
    while (sum < v && it.hasNext()) {
      result = it.next();
      sum += slot;
    }
    return new Experiment[]{result};
  }

  @Override
  public void feedback(User user, Query query, Experiment[] config, double score) {
    for (final Experiment exp : config) {
      Stat stat = value.get(exp);
      if (stat == null)
        value.put(exp, stat = new Stat());
      stat.update(score);
    }
  }

  @Override
  public Stat score(Experiment experiment) {
    final Stat stat = value.get(experiment);
    return stat != null ? stat : new Stat();
  }
}
