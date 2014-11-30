package com.spbsu.ml.methods.greedyRegion.hierarchicalCherryRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Created by noxoomo on 24/11/14.
 */
//TODO: RegionStats merge vs RegionLayer (add lazy points initialization)
public class HierarchicalPick {
  private final BinarizedDataSet bds;
  private final BFGrid grid;
  private final Factory<AdditiveStatistics> factory;
  private final int[][] binsIndex;
  public double currentScore = 0;
  final int binsCount;
  final int take2grow;
  final CategoricalAggregate aggregate;
  double lambda;
  final int[][] useCounts;

  public HierarchicalPick(BinarizedDataSet bds, Factory<AdditiveStatistics> factory, int take2grow, double lambda, int[][] useCounts) {
    this.lambda = lambda;
    this.bds = bds;
    this.grid = bds.grid();
    this.factory = factory;
    this.binsIndex = new int[grid.rows()][];
    this.useCounts = useCounts;

    int current = 0;
    for (int i = 0; i < grid.rows(); i++) {
      this.binsIndex[i] = new int[grid.row(i).size() + 1];
      for (int bin = 0; bin <= grid.row(i).size(); ++bin) {
        this.binsIndex[i][bin] = current;
        ++current;
      }
    }
    this.binsCount = current;
    aggregate = new CategoricalAggregate(bds, factory);
    this.take2grow = take2grow;
  }


  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Pick thread", -1);


  BitSet used = new BitSet();

  public <T extends AdditiveStatistics> RegionLayer<T> build(final Evaluator<T> eval, int[] points) {
    if (points == null || points.length == 0) {
      throw  new RuntimeException("error");
    }

    final ModelComplexityCalcer modelComplexity = new ModelComplexityCalcer(bds, binsIndex, points);
    ArrayList<RegionLayer<T>> models = init(eval, points, modelComplexity);

    RegionLayer resultModel = chooseBest(models,eval);

    do {
      ArrayList<RegionStats<T>> fittedModels = new ArrayList<>(models.size() * (1 + binsCount));

      for (RegionLayer layer : models) {
        final RegionLayer<T> finalLayer = layer;
        final RegionStats[] buildResult = build(new Evaluator<T>() {
          @Override
          public double value(T stat) {
            T totalStat = (T) factory.create();
            totalStat.append(stat);
            totalStat.append(finalLayer.inside);
            return eval.value(totalStat);
          }
        }, layer.conditions, layer.outsidePoints, modelComplexity);

        for (RegionStats stat : buildResult) {
          stat.basedOn = layer;
          if (stat.information < lambda)
            fittedModels.add(stat);
        }
      }

      models = takeGrowPart(fittedModels);
      RegionLayer bestModel = chooseBest(models,eval);
      bestModel = chooseBest(bestModel,resultModel,eval);
      if (bestModel == resultModel || bestModel.score == Double.POSITIVE_INFINITY) {
        break;
      }
      resultModel = bestModel;
    } while (true);

    used.or(resultModel.conditions);
    return resultModel;
  }

  private <T extends AdditiveStatistics> RegionLayer chooseBest(ArrayList<RegionLayer<T>> models, Evaluator<T> eval) {
    RegionLayer resultModel = models.get(0);
    for (int i = 0; i < models.size(); ++i)
      resultModel = chooseBest(resultModel, models.get(i),eval);
    return resultModel;
  }

  private <T extends AdditiveStatistics> ArrayList<RegionLayer<T>> init(final Evaluator<T> eval, int[] points, ModelComplexityCalcer calcer) {
    RegionLayer fakeLayer = new RegionLayer(factory.create(), new BitSet(binsCount), Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, new int[0], points);
    final RegionStats[] buildResult = build(eval, new BitSet(binsCount), points, calcer);
    ArrayList<RegionStats<T>> fittedModels = new ArrayList<>(buildResult.length);
    for (RegionStats stat : buildResult) {
      if (stat.information < lambda) {
        stat.basedOn = fakeLayer;
        fittedModels.add(stat);
      }
    }
    return takeGrowPart(fittedModels);
  }

  private <T extends AdditiveStatistics> ArrayList<RegionLayer<T>> takeGrowPart(ArrayList<RegionStats<T>> models) {
    ArrayList<RegionLayer<T>> result = new ArrayList<>(take2grow);

    if (take2grow >= models.size()) {
      for (int i = 0; i < models.size(); ++i) {
        result.add(split(models.get(i)));
      }
      return result;
    }

    Collections.sort(models, new Comparator<RegionStats<T>>() {
      @Override
      public int compare(RegionStats<T> o1, RegionStats<T> o2) {
        if (o1.score == o2.score) {
          return Double.compare(o1.information, o2.information);
        }
        return Double.compare(o1.score, o2.score);
      }
    });

    for (int i = 0; i < take2grow; ++i) {
      result.add(split(models.get(i)));
    }
    return result;
  }

  private <T extends AdditiveStatistics> RegionLayer<T> split(RegionStats<T> best) {
    int[] points = best.basedOn.outsidePoints;
    T inside = (T) factory.create();
    inside.append(best.basedOn.inside);
    TIntArrayList included = new TIntArrayList();
    TIntArrayList excluded = new TIntArrayList();
    byte[] bins = bds.bins(best.feature);
    for (int point : points) {
      if (bins[point] == best.bin) {
        included.add(point);
        inside.append(point, 1);
      } else {
        excluded.add(point);
      }
    }
    included.add(best.basedOn.insidePoints);
    BitSet conditions = (BitSet) best.basedOn.conditions.clone();
    conditions.set(binsIndex[best.feature][best.bin]);
    return new RegionLayer<>(inside, conditions, best.information, best.score, included.toArray(), excluded.toArray());
  }


  private RegionLayer chooseBest(RegionLayer bestModel, RegionLayer model,final Evaluator eval) {
    if (bestModel == null) {
      return model;
    }
    if (model.information == Double.POSITIVE_INFINITY) {
      return bestModel;
    }

    if (eval.value(bestModel.inside) <= eval.value(model.inside) + 1e-9) {
      return bestModel;
    }
    return model;
  }


  @SuppressWarnings("unchecked")
  public <T extends AdditiveStatistics> RegionStats<T>[] build(final Evaluator<T> eval, final BitSet condition, final int[] points, final ModelComplexityCalcer modelComplexity) {
    final RegionStats<T> result[] = new RegionStats[binsCount - condition.cardinality()];
    final CountDownLatch latch = new CountDownLatch(result.length);
    final T total = (T) factory.create();
    for (int point : points)
      total.append(point, 1);

    int index = 0;
    for (int f = 0; f < grid.rows(); ++f) {
      final int finalFeature = f;
      for (int bin = 0; bin <= grid.row(f).size(); ++bin) {
        if (condition.get(binsIndex[f][bin])) {
          continue;
        }

        if (used.get(binsIndex[f][bin])) { //skip features, used on previous layers
          result[index] = new RegionStats<>((T) factory.create(), finalFeature, bin, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
          latch.countDown();
          ++index;
          continue;
        }

        final int finalIndex = index;
        final int finalBin = bin;
        ++index;
        condition.set(binsIndex[f][bin]);
        final double complexity = modelComplexity.calculate(condition);
        condition.clear(binsIndex[f][bin]);

        exec.submit(new Runnable() {
          @Override
          public void run() {
            final byte[] bin = bds.bins(finalFeature);
            final T stat = (T) factory.create();
            for (int i = 0; i < points.length; i++) {
              final int index = points[i];
              if (finalBin == bin[index])
                stat.append(index, 1);
            }
//            final double score = eval.value(stat) * (1.0 / (1 + Math.log(useCounts[finalFeature][finalBin] + 1))) * (1.0 / (1 + 0.005 * complexity));
            final double score = eval.value(stat);// * (1.0 / (1 + Math.log(useCounts[finalFeature][finalBin] + 1))) * (1.0 / (1 + 0.005 * complexity));
            result[finalIndex] = new RegionStats<T>(stat, finalFeature, finalBin, complexity, score);
            latch.countDown();
          }
        });
      }
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
    }
    return result;
  }


  class ModelComplexityCalcer {
    private final BFGrid grid;
    private final int[][] binsIndex;
    private final int[][] base;
    public double total;

    public ModelComplexityCalcer(BinarizedDataSet bds, int[][] binsIndex, int[] points) {
      this.grid = bds.grid();
      this.binsIndex = binsIndex;
      base = new int[grid.rows()][];
      {
        for (int feature = 0; feature < grid.rows(); feature++) {
          base[feature] = new int[grid.row(feature).size() + 1];
          final byte[] bin = bds.bins(feature);
          for (int j = 0; j < points.length; j++) {
            base[feature][bin[points[j]]]++;
          }
        }
      }
      total = 0;
      for (int bin = 0; bin <= grid.row(0).size(); ++bin) {
        total += base[0][bin];
      }
    }

    public double calculate(BitSet conditions) {
      double bits = 0;
      int usedFeatures = 0;
      for (int f = 0; f < grid.rows(); ++f) {
        boolean used = false;
        double entropy = 0;
        int realCardinality = 0;
        double count = 0;
        boolean current = false;
        for (int bin = 0; bin <= grid.row(f).size(); ++bin) {
          if (conditions.get(binsIndex[f][bin]) == current) {
            count += base[f][bin];
          } else {
            realCardinality++;
            entropy += count > 0 ? count * Math.log(count) : 0;
            current = conditions.get(binsIndex[f][bin]);
            used = current ? current : used;
            count = base[f][bin];
          }
          if (current && binsIndex[f][bin] == 0)
            return Double.POSITIVE_INFINITY;
        }
        if (!used)
          continue;
        ++usedFeatures;
        realCardinality++;
        entropy += count > 0 ? count * Math.log(count) : 0;
        entropy /= total;
        entropy = Math.log(total) - entropy;
        bits += realCardinality;
        bits += entropy;
      }
      bits += usedFeatures * Math.log(grid.rows());
      return bits;
    }
  }
}


