package com.spbsu.exp.modelexp;

import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.exp.modelexp.managers.SerpManager;
import com.spbsu.exp.modelexp.setup.SimpleExclusive;
import com.spbsu.exp.modelexp.users.UserFactory;
import gnu.trove.map.TObjectDoubleMap;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * User: solar
 * Date: 02.04.15
 * Time: 19:18
 */
public class Model {
  private final int initialUsers;
  private final int days;
  private final double pBirth;

  private final FastRandom rng;
  private List<Experiment> experiments = new ArrayList<>();

  public Model(int initialUsers, int days, double pBirth, FastRandom rng) {
    this.initialUsers = initialUsers;
    this.days = days;
    this.pBirth = pBirth;
    this.rng = rng;
  }

  @SafeVarargs
  public final double model(Setup setup, Factory<User> usersFactory, Factory<Experiment[]>... managers) {
    final List<User> usersPool = new ArrayList<>(20000000);

    for (int i = 0; i < initialUsers; i++) {
      usersPool.add(usersFactory.create());
    }

    double result = 0;
    for (int day = 0; day < days; day++) {
      final TObjectDoubleMap<Experiment> totalScore = new TObjectDoubleHashMap<>();

      // daily manager needs
      for(int i = 0; i < managers.length; i++) {
        final Factory<Experiment[]> manager = managers[i];
        final Experiment[] experiments = manager.create();
        for (Experiment experiment : experiments) {
          setup.add(experiment);
          this.experiments.add(experiment);
        }
      }

      // production change
      setup.nextDay();
      double dayMeanScore = 0;
      int userActivity = 0;
      for (int hour = 0; hour < 24; hour++) {
        final int newborn = rng.nextPoisson(pBirth / 24 * usersPool.size());
        for(int i = 0; i < newborn; i++) {
          usersPool.add(usersFactory.create());
        }
        final RandSet<User> userRandSet = new RandSet<>(usersPool.toArray(new User[usersPool.size()]), new Evaluator<User>() {
          @Override
          public double value(User user) {
            return user.activity() / 24;
          }
        }, rng);

        final int hourActivity = rng.nextPoisson(userRandSet.total());
        userActivity += hourActivity;
        for(int i = 0; i < hourActivity; i++) {
          final User user = userRandSet.next();
          final Query query = user.next(day, hour);
          final Experiment[] config = setup.assign(user, query);
          double score = 0;
          for(int j = 0; j < config.length; j++) {
            if (config[j].relevant(query))
              score += config[j].work(query);
          }
          result += score;
          dayMeanScore += score / 24 / hourActivity;
          setup.feedback(user, query, config, score);
          user.feedback(score);
          for (final Experiment experiment : config) {
            totalScore.adjustOrPutValue(experiment, score, score);
          }
        }
      }
      System.out.println("Day " + day + " score: " + dayMeanScore + " user activity: " + userActivity);
    }
    return result;
  }

  private Experiment[] experiments() {
    return experiments.toArray(new Experiment[experiments.size()]);
  }

  public static void main(String[] args) {
    final FastRandom rng = new FastRandom(0);
    final Model model = new Model(100000, 365, 0.01, rng);
    final Setup setup = new SimpleExclusive(rng);
    final double score = model.model(setup, new UserFactory(rng), new SerpManager(rng));

    TObjectIntMap<Pair<Stat.Verdict, Stat.Verdict>> results = new TObjectIntHashMap<>();
    for (Experiment experiment : model.experiments()) {
      final Pair<Stat.Verdict, Stat.Verdict> cell = Pair.create(experiment.realScore() > 0 ? Stat.Verdict.GOOD : Stat.Verdict.BAD, setup.score(experiment).status());
      results.adjustOrPutValue(cell, 1, 1);
    }

    System.out.println("Total user experience: " + score);
    for (Stat.Verdict verdict : Stat.Verdict.values())
      System.out.print("\t\t" + verdict);
    System.out.println();
    for (Stat.Verdict v1 : Stat.Verdict.values()) {
      System.out.print(v1);
      for (Stat.Verdict v2 : Stat.Verdict.values()) {
        System.out.print("\t" + results.get(Pair.create(v1, v2)));
      }
      System.out.println();
    }
  }

  public class RandSet<T> {
    private final T[] arr;
    private final FastRandom rng;
    private final double[] weights;
    private final double sum;

    public RandSet(T[] arr, Evaluator<T> eval, FastRandom rng) {
      this.arr = arr;
      this.rng = rng;
      this.weights = new double[arr.length];
      double sum = 0;
      for(int i = 0; i < arr.length; i++) {
        sum += eval.value(arr[i]);
        weights[i] = sum;
      }
      this.sum = sum;
    }

    public T next() {
      final double v = rng.nextDouble() * sum;
      int index = Arrays.binarySearch(weights, v);
      index = index > 0 ? index : -(index + 1);
//      System.out.println(index > 0 ? weights[index] - weights[index - 1] : weights[index]);
      return arr[index];
    }

    public double total() {
      return sum;
    }
  }
}
