package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Holder;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.models.Region;

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
public class GreedyRegion implements Optimization<WeightedLoss<L2>> {
  public static final int NN_NEIGHBORHOOD = 1000;
  private final Random rng;
  private final BFGrid grid;
  byte[][] binarization;
  int[] nn;
  private double alpha = 10;
  private double betta = 0.00001;

  public GreedyRegion(Random rng, BFGrid grid) {
    this.rng = rng;
    this.grid = grid;
  }

  private void prepareNN(DataSet ds) {
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
  ThreadPoolExecutor exec = new ThreadPoolExecutor(POOL_SIZE, POOL_SIZE, 1, TimeUnit.SECONDS, new ArrayBlockingQueue<Runnable>(100));
  @Override
  public synchronized Region fit(final DataSet learn, final WeightedLoss<L2> loss) {
    prepareNN(learn);
    final Holder<Region> answer = new Holder<Region>(null);
    final CountDownLatch latch = new CountDownLatch(POOL_SIZE);
    for (int i = 0; i < Runtime.getRuntime().availableProcessors(); i++) {
      exec.execute(new Runnable() {
        @Override
        public void run() {
          final Region model = fitInner(learn, loss);
          synchronized (answer) {
            if (answer.getValue() == null || answer.getValue().score() > model.score())
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

  public Region fitInner(DataSet learn, StatBasedLoss loss) {
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
//      if (rng.nextDouble() > 100. / grid.size())
      conditions.add(bc);
    }

    List<BinaryCond> best = null;
    double bestMean = 0.;
    double bestScore = Double.MAX_VALUE;
    int bestCount = 0;
    while (!conditions.isEmpty()) {
      final int currentConditionsCount = conditions.size();
      double[] csum = new double[currentConditionsCount];
      double[] csum2 = new double[currentConditionsCount];
      int[] ccount = new int[currentConditionsCount];
      double sum = 0;
      double sum2 = 0;
      int count = 0;
      for (int t = 0; t < order.length; t++) {
        if (l1dist[t] > grid.size() - currentConditionsCount - 1)
          break;
        int matches = currentConditionsCount;
        int lastUnmatch = 0;
        final int index = order[t];
        byte[] currentFolds = binarization[index];
        for (int i = 0; i < currentConditionsCount && matches > currentConditionsCount - 2; i++) {
          BinaryCond next = conditions.get(i);
          if (!next.yes(currentFolds)) {
            matches--;
            lastUnmatch = i;
          }
        }
        if (matches == currentConditionsCount) {
          final double y = learn.target().get(index);
          sum += y;
          sum2 += y * y;
          count++;
        }
        else if (matches == currentConditionsCount - 1) { // the only binary feature has not matched
          final double y = learn.target().get(index);
          csum[lastUnmatch] += y;
          csum2[lastUnmatch] += y * y;
          ccount[lastUnmatch]++;
        }
      }
      { // best region update
        double score = score(total, count, sum, sum2, currentConditionsCount);
        if (score < bestScore) {
          best = new ArrayList<BinaryCond>(conditions);
          bestScore = score;
          bestMean = sum/count;
          bestCount = count;
        }
      }
      { // choose what condition should be dropped
        int worst = (int)(currentConditionsCount * rng.nextDouble());
        double score = score(total, count + ccount[worst], sum + csum[worst], sum2 + csum2[worst], currentConditionsCount - 1);
        for (int i = 0; i < currentConditionsCount; i++) {
          double cscore = score(total, count + ccount[i], sum + csum[i], sum2 + csum2[i], currentConditionsCount - 1);
          if (cscore < score) {
            worst = i;
            score = cscore;
          }
        }
        conditions.remove(worst);
      }
    }
    List<BFGrid.BinaryFeature> features = new ArrayList<BFGrid.BinaryFeature>();
    boolean[] mask = new boolean[best.size()];
    for (int i = 0; i < best.size(); i++) {
      features.add(best.get(i).bf);
      mask[i] = best.get(i).mask;
    }

    return new Region(features, mask, bestMean, bestCount, bestScore);
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
}
