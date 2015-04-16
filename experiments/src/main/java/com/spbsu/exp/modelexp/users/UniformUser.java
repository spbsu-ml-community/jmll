package com.spbsu.exp.modelexp.users;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.exp.modelexp.Query;
import com.spbsu.exp.modelexp.User;

import java.util.Arrays;

/**
* User: solar
* Date: 07.04.15
* Time: 14:43
*/
public class UniformUser implements User {
  protected final FastRandom rng;
  private final double lambda;
  private final short[] intents;
  private final static int[][] stats = new int[Query.INTENTS_LEN][65536];
//  private static Vec intentWeights;
//
//  static {
//    intentWeights = new ArrayVec(INTENTS_LEN);
//    for (int i = 0; i < intentWeights.length(); i++) {
//      intentWeights.set(i, Math.pow(1, i));
//    }
//  }

  public UniformUser(FastRandom rng, double lambda) {
    this.rng = rng;
    this.lambda = lambda;
    this.intents = new short[Query.INTENTS_LEN];
    for(int i = 0; i < Query.INTENTS_LEN; i++) {
      final int scale = 1 + 2 * i;
      final double gamma = rng.nextGamma(1, scale);
      final double v = 1 + gamma;
      if (v > 32768)
        throw new RuntimeException();
      final short next = (short) v;
      intents[i] = next;
      stats[i][next]++;
    }
  }

  public short[] properties() {
    return intents;
  }

  @Override
  public Query next(int day, int hour) {
    final short[] qIntents = new short[Query.INTENTS_LEN];
    int intents = rng.nextPoisson(2) + 1;
    for (int i = 0; i < intents; i++) {
      final int intent = (int) (rng.nextDouble() * Query.INTENTS_LEN);
      qIntents[intent] = this.intents[intent];
    }

    final IntentBasedQuery query = new IntentBasedQuery(qIntents);
//    freqs.adjustOrPutValue(query, 1, 1);
//    boolean val = freqs.size() == 1000000;
//    if (val) {
//      final Object[] keys = freqs.keys();
//      Arrays.sort(keys, new Comparator<Object>() {
//        @Override
//        public int compare(Object o1, Object o2) {
//          return Integer.compare(freqs.get(o2), freqs.get(o1));
//        }
//      });
//      for(int i = 0; i < keys.length && i < 100; i++) {
//        System.out.println(freqs.get(keys[i]) + " -> " + keys[i]);
//      }
//      System.out.println();
//    }

    return query;
  }

  @Override
  public void feedback(double score) {
  }

  @Override
  public double activity() {
    return lambda;
  }

  private class IntentBasedQuery implements Query {
    private final short[] qIntents;

    public IntentBasedQuery(short[] qIntents) {
      this.qIntents = qIntents;
    }

    @Override
    public short[] properties() {
      return qIntents;
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(qIntents);
    }

    @Override
    public String toString() {
      return Arrays.toString(qIntents);
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof IntentBasedQuery && Arrays.equals(qIntents, ((IntentBasedQuery) obj).qIntents);
    }
  }
}
