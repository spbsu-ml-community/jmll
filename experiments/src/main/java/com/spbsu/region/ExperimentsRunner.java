package com.spbsu.region;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.GridTools;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.L2GreedyTDRegion;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.greedyRegion.RegionForest;
import gnu.trove.list.array.TDoubleArrayList;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Created by noxoomo on 11/07/14.
 */

class ScoresCalcer {

  static FastRandom random = new FastRandom(10);
  public int iterations = 12000;
  public double step = 0.004;
  private Pool<?> learn;
  private Pool<?> validate;
  static ThreadPoolExecutor pool = ThreadTools.createBGExecutor("Boosting thread", -1);

  public ScoresCalcer(Pool<?> learn, Pool<?> validate) {
    this.learn = learn;
    this.validate = validate;
  }


  public double[] run(int tries) {
    //alpha 0.075652  and beta 1.164409 s
    return run(tries, 0.02, 0.5);
  }

  public double[] run(int tries, final double alpha, final double beta) {
    final double[] scores = new double[tries];
    final CountDownLatch latch = new CountDownLatch(tries);
    for (int i = 0; i < tries; ++i) {
      final long seed = random.nextLong();
      final int index = i;
      pool.execute(new Runnable() {
        @Override
        public void run() {
          final GradientBoosting<L2> boosting = new GradientBoosting
                  (new RegionForest<>(GridTools.medianGrid(learn.vecData(), 32), new FastRandom(seed), 5, alpha, beta), L2GreedyTDRegion.class, iterations, step);
//          new GradientBoosting
//                  (new BootstrapOptimization(
//                          new GreedyTDWeakRegion2<>(GridTools.medianGrid(learn.vecData(), 32), alpha, beta), new FastRandom(seed)), L2GreedyRegion.class, iterations, step);
//                          new GreedyTDWeakRegion2<>(GridTools.medianGrid(learn.vecData(), 32)), new FastRandom(seed),alpha,beta), L2GreedyRegion.class, iterations, step);

          final ScoreCalcer validateListener = new ScoreCalcer(/*"\ttest:\t"*/"\t", validate.vecData(), validate.target(L2.class));
          boosting.addListener(validateListener);
          final Ensemble ensemble = boosting.fit(learn.vecData(), learn.target(L2.class));
          scores[index] = validateListener.min;
          System.out.println("Score for run " + validateListener.min);
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    return scores;
  }


  protected static class ScoreCalcer implements ProgressHandler {
    final String message;
    final Vec current;
    private final VecDataSet ds;
    private final L2 target;

    public ScoreCalcer(String message, VecDataSet ds, L2 target) {
      this.message = message;
      this.ds = ds;
      this.target = target;
      current = new ArrayVec(ds.length());
    }

    double min = 1e10;

    @Override
    public void invoke(Trans partial) {
      if (partial instanceof Ensemble) {
        final Ensemble linear = (Ensemble) partial;
        final Trans increment = linear.last();
        for (int i = 0; i < ds.length(); i++) {
          if (increment instanceof Ensemble) {
            current.adjust(i, linear.wlast() * (increment.trans(ds.data().row(i)).get(0)));
          } else {
            current.adjust(i, linear.wlast() * ((Func) increment).value(ds.data().row(i)));
          }
        }
      } else {
        for (int i = 0; i < ds.length(); i++) {
          current.set(i, ((Func) partial).value(ds.data().row(i)));
        }
      }
      double curLoss = VecTools.distance(current, target.target) / Math.sqrt(ds.length());
//      System.out.print(message + curLoss);
      min = Math.min(curLoss, min);
//      System.out.print(" minimum = " + min);
    }
  }


}

public class ExperimentsRunner {
  static FastRandom random = new FastRandom(0);
  private static Executor pool = ThreadTools.createBGExecutor("Boosting thread", -1);


  public static void main(String[] args) {
    try {
      String learnPath = "features.txt.gz";
      String testPath = "featuresTest.txt.gz";
      final Pool<?> learn = DataTools.loadFromFeaturesTxt(learnPath);
      final Pool<?> validate = DataTools.loadFromFeaturesTxt(testPath);
      System.err.println(String.format("Learn size %d\nValidation size: %d", learn.data().length(), validate.data().length()));


      switch (args[0]) {
        case "Region": {
          ScoresCalcer calcer = new ScoresCalcer(learn, validate);
          final double[] scores = calcer.run(16);
          System.out.println(String.format("Min scores for runs with parametrs iter = %d, step = %f", calcer.iterations, calcer.step));
          System.out.println(Arrays.toString(scores));
          break;
        }

        case "RandomSearch": {
          int tries = 200;
          final int bootstrapRuns = 5;
          final double[] meanScores = new double[tries];
          final double[] minScores = new double[tries];
          final double[] maxScores = new double[tries];
          final double[] alphas = new double[tries];
          final double[] betas = new double[tries];

          final CountDownLatch latch = new CountDownLatch(tries);

          for (int i = 0; i < tries; ++i) {
            final int index = i;
            alphas[i] = random.nextDouble() * 2;
            betas[i] = random.nextDouble() * 2;
            pool.execute(new Runnable() {
              @Override
              public void run() {
                final ScoresCalcer calcer = new ScoresCalcer(learn, validate);
                final double[] scores = calcer.run(bootstrapRuns, alphas[index], betas[index]);
                final double[] stat = stats(scores);
                String msg = String.format("For alpha %f  and beta %f scores are :\n", +alphas[index], betas[index]);
                System.out.println(msg + Arrays.toString(scores));
                meanScores[index] = stat[0];
                minScores[index] = stat[1];
                maxScores[index] = stat[2];
                latch.countDown();
              }
            });
          }
          try {
            latch.await();
          } catch (InterruptedException e) {
            e.printStackTrace();
          }

          System.out.println(String.format("alphas: "));
          System.out.println(Arrays.toString(alphas));

          System.out.println(String.format("betas: "));
          System.out.println(Arrays.toString(betas));

          System.out.println(String.format("meanScores: "));
          System.out.println(Arrays.toString(meanScores));

          System.out.println(String.format("minScores: "));
          System.out.println(Arrays.toString(minScores));

          System.out.println(String.format("maxcores: "));
          System.out.println(Arrays.toString(maxScores));
          break;
        }
      }

    } catch (IOException e) {
      e.printStackTrace();
    }
  }


  public static double[] stats(double[] sample) {
    if (sample.length == 0) {
      return new double[3];
    }
    double min;
    double max;
    min = max = sample[0];
    double mean = 0;
    double result[] = new double[3];
    for (double score : sample) {
      min = min < score ? min : score;
      max = max > score ? max : score;
      mean += score;
    }

    mean /= sample.length;
    result[0] = mean;
    result[1] = min;
    result[2] = max;
    return result;
  }


  public static double mean(TDoubleArrayList sample) {
    double mean = 0;
    for (int i = 0; i < sample.size(); ++i)
      mean += sample.get(i);
    return mean /= sample.size();
  }

  public static double var(TDoubleArrayList sample) {
    double secondMoment = 0;
    for (int i = 0; i < sample.size(); ++i) {
      double d = sample.get(i);
      secondMoment += d * d;
    }
    secondMoment /= sample.size();
    double m = mean(sample);
    return secondMoment - m * m;
  }

  public static double[] stats(TDoubleArrayList sample) {
    if (sample.size() == 0) {
      return new double[4];
    }
    double min;
    double max;
    min = max = sample.get(0);
    double mean = 0;
    double result[] = new double[4];
    double secondMoment = 0;
    for (int i = 0; i < sample.size(); ++i) {
      double d = sample.get(i);
      secondMoment += d * d;
      mean += d;
      if (d > max) max = d;
      if (d < min) min = d;
    }
    mean /= sample.size();
    secondMoment /= sample.size();
    result[0] = mean;
    result[1] = secondMoment - mean * mean;
    result[1] = Math.sqrt(result[1]);
    result[2] = min;
    result[3] = max;
    return result;
  }


}



