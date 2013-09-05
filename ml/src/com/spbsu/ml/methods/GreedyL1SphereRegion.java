package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Holder;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.L2Loss;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyL1SphereRegion implements MLMethodOrder1 {
  public static final int NN_NEIGHBORHOOD = 100;
  private final Random rng;
  private final BFGrid grid;
  byte[][] binarization;
  int[] nn;
  private double alpha = 10;
  private double betta = 0.00001;

  public GreedyL1SphereRegion(Random rng, DataSet ds, BFGrid grid) {
    this.rng = rng;
    this.grid = grid;
    final int total = ds.power();
    binarization = new byte[total][];
    final DSIterator iter = ds.iterator();
    int index = 0;
    while (iter.advance()) {
      final byte[] folds = binarization[index++] = new byte[iter.x().dim()];
      grid.binarize(iter.x(), folds);
    }
    nn = new int[total * NN_NEIGHBORHOOD];
    int[] l1dist = new int[total];
    for (int i = 0; i < total; i++) {
      byte[] folds = binarization[i];
      int[] order = ArrayTools.sequence(0, total);
      {
        for (int t = 0; t < binarization.length; t++) {
          final byte[] currentFolds = binarization[t];
          int l1 = 0;
          for (int f = 0; f < folds.length; f++) {
            int diff = folds[f] - currentFolds[f];
            l1 += diff > 0 ? diff : -diff;
          }
          l1dist[t] = l1;
        }
        ArrayTools.parallelSort(l1dist, order);
        for (int t = 0; t < NN_NEIGHBORHOOD; t++) {
          nn[i * NN_NEIGHBORHOOD + t] = order[t];
        }
      }
    }
  }

  public static final int POOL_SIZE = Runtime.getRuntime().availableProcessors();
  ThreadPoolExecutor exec = new ThreadPoolExecutor(POOL_SIZE, POOL_SIZE, 100500, TimeUnit.DAYS, new ArrayBlockingQueue<Runnable>(100));

  @Override
  public Model fit(DataSet learn, Oracle1 loss) {
    return fit(learn, loss, new ArrayVec(learn.power()));
  }

  @Override
  public Region fit(final DataSet learn, final Oracle1 loss, final Vec start) {
    final Holder<Region> answer = new Holder<Region>(null);
    final CountDownLatch latch = new CountDownLatch(POOL_SIZE);
    for (int i = 0; i < Runtime.getRuntime().availableProcessors(); i++) {
      exec.execute(new Runnable() {
        @Override
        public void run() {
          final Region model = fitInner(learn, loss, start);
          synchronized (answer) {
            if (answer.getValue() == null || answer.getValue().score > model.score)
              answer.setValue(model);
          }
          latch.countDown();
        }
      });
    }

    try {
      latch.await();
    } catch (InterruptedException e) {
      // skip
    }
    return answer.getValue();
  }

  public Region fitInner(DataSet ds, Oracle1 loss, Vec start) {
    DataSet learn = ds;
    assert loss.getClass() == L2Loss.class;
    int pointIdx = choosePointAtRandomNN(learn);

    byte[] folds = binarization[pointIdx];
    final int total = learn.power();
    int[] order = ArrayTools.sequence(0, total);
    int[] l1dist = new int[total];
    {
      for (int i = 0; i < binarization.length; i++) {
        final byte[] currentFolds = binarization[i];
        int l1 = 0;
        for (int f = 0; f < folds.length; f++) {
          int diff = folds[f] - currentFolds[f];
          l1 += diff > 0 ? diff : -diff;
        }
        l1dist[i] = l1;
      }
      ArrayTools.parallelSort(l1dist, order);
    }

    final List<BinaryCond> conditions = new ArrayList<BinaryCond>(grid.size());
    for (int bf = 0; bf < grid.size(); bf++) {
      BinaryCond bc = new BinaryCond();
      bc.bf = grid.bf(bf);
      bc.mask = bc.bf.value(folds);
      conditions.add(bc);
    }

    double bestMean = 0.;
    double bestScore = Double.MAX_VALUE;
    int bestCount = 0;
    int bestL1Dist = 0;
    double sum = 0;
    double sum2 = 0;
    for (int t = 0; t < order.length; t++) {
      final int index = order[t];
      double y = learn.target().get(index);
      sum += y;
      sum2 += y * y;
      double score = score(total, t, sum, sum2, 0);
      if (score <= bestScore) {
        bestScore = score;
        bestMean = sum/t;
        bestCount = t;
        bestL1Dist = l1dist[t];
      }
    }
    return new Region(conditions, bestMean, bestL1Dist, bestCount, bestScore);
  }

  private double score(int total, int count, double sum, double sum2, int ccount) {
    final double err = -sum * sum / count;
    return err * (1. - 2 * (Math.log(2)/ Math.log(count + 1.) + (total > count ? Math.log(2)/ Math.log(total - count + 1.) : 0))) + betta * ccount;
  }

  private int choosePointAtRandom(DataSet learn) {
    double total = 0.;
    {
      final DSIterator dsIterator = learn.iterator();
      while (dsIterator.advance()) {
        total += Math.abs(dsIterator.y());
      }
    }
    int startPointIndex = 0;
    {
      double rnd = total * rng.nextDouble();
      final DSIterator dsIterator = learn.iterator();
      while (dsIterator.advance() && rnd > Math.abs(dsIterator.y())) {
        rnd -= Math.abs(dsIterator.y());
        startPointIndex++;
      }
    }
    return startPointIndex;
  }

  private int choosePointAtRandomNN(DataSet learn) {
    double total = 0.;
    double[] weights = new double[learn.power()];
    double max = 0;
    int maxIndex = -1;
    final Vec target = learn.target();
    for (int i = 0; i < weights.length; i++) {
      double sum = 0;
      for (int t = 0; t < NN_NEIGHBORHOOD; t++) {
        sum += target.get(nn[i * NN_NEIGHBORHOOD + t]);
      }
      weights[i] = sum * sum;
      total += weights[i];
      if (max < weights[i]) {
        max = weights[i];
        maxIndex = i;
      }
    }
//    return maxIndex;

    double rnd = total * rng.nextDouble();
    for (int i = 0; i < weights.length; i++) {
      if (rnd <= weights[i])
        return i;
      rnd -= weights[i];
    }
    return 0;
  }

  private static class BinaryCond {
    BFGrid.BinaryFeature bf;
    boolean mask;

    public boolean yes(byte[] folds) {
      return bf.value(folds) == mask;
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append(" ")
              .append(bf.findex)
              .append(mask ? ">=" : "<")
              .append(bf.condition);

      return builder.toString();
    }
  }

  public static class Region extends Model {
    private final int[] features;
    private final double[] conditions;
    private final boolean[] mask;
    private final double value;
    private final int basedOn;
    private final double score;
    private final int dist;

    public Region(final List<BinaryCond> conditions, double value, int dist, int basedOn, double bestScore) {
      this.dist = dist;
      this.basedOn = basedOn;
      this.features = new int[conditions.size()];
      this.conditions = new double[conditions.size()];
      this.mask = new boolean[conditions.size()];
      this.value = value;
      for (int i = 0; i < conditions.size(); i++) {
        this.features[i] = conditions.get(i).bf.findex;
        this.conditions[i] = conditions.get(i).bf.condition;
        this.mask[i] = conditions.get(i).mask;
      }
      this.score = bestScore;
    }

    @Override
    public double value(Vec x) {
      int matches = 0;
      for (int i = 0; i < features.length; i++) {
        if ((x.get(features[i]) >= conditions[i]) == mask[i])
          matches++;
      }
      return (features.length - matches) <= dist ? value : 0;
    }

    public boolean contains(Vec x) {
      for (int i = 0; i < features.length; i++) {
        if ((x.get(features[i]) >= conditions[i]) != mask[i])
          return false;
      }

      return true;
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append(value).append("/").append(basedOn);
      builder.append(" ->");
      for (int i = 0; i < features.length; i++) {
        builder.append(" ")
               .append(features[i])
               .append(mask[i] ? ">=" : "<")
               .append(conditions[i]);
      }
      return builder.toString();
    }
  }
}
