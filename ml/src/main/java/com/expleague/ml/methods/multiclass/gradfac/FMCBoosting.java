package com.expleague.ml.methods.multiclass.gradfac;

import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.factorization.Factorization;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.ScaledVectorFunc;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.ObliviousTree;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Experts League Created by solar on 05.05.17.
 */
public class FMCBoosting extends WeakListenerHolderImpl<Trans> {
  protected final VecOptimization<StatBasedLoss> weak;
  private final Class<? extends L2> factory;
  private final Factorization factorize;
  private final int iterationsCount;
  private double step;
  private final int ensembleSize;
  private final boolean isGbdt;
  private BinarizedDataSet bds = null;
  private final FastRandom rfRnd = new FastRandom(13);

  public FMCBoosting(final Factorization factorize, final VecOptimization<StatBasedLoss> weak,
                     final int iterationsCount, final double step) {
    this(factorize, weak, SatL2.class, iterationsCount, step, 1, false);
  }

  public FMCBoosting(final Factorization factorize, final VecOptimization<StatBasedLoss> weak,
                     final int iterationsCount, final double step, final int ensembleSize) {
    this(factorize, weak, SatL2.class, iterationsCount, step, ensembleSize, false);
  }

  public FMCBoosting(final Factorization factorize, final VecOptimization<StatBasedLoss> weak,
                     final Class<? extends L2> factory, final int iterationsCount, final double step) {
    this(factorize, weak, factory, iterationsCount, step, 1, false);
  }

  public FMCBoosting(Factorization factorize, final VecOptimization<StatBasedLoss> weak,
                     final Class<? extends L2> factory, final int iterationsCount, final double step,
                     final int ensembleSize, final boolean isGbdt) {
    this.factorize = factorize;
    this.weak = weak;
    this.factory = factory;
    this.iterationsCount = iterationsCount;
    this.step = step;
    this.ensembleSize = ensembleSize;
    this.isGbdt = isGbdt;
  }

  public Ensemble<ScaledVectorFunc> fit(Pool<?> trainPool) {
    final VecDataSet learn = trainPool.vecData();
    final Mx probabilities = getMultiLabelTarget(trainPool);

    // Distribute probability across labels
    for (int i = 0; i < probabilities.rows(); ++i) {
      VecTools.normalizeL1(probabilities.row(i));
    }

    final Vec[] B = new Vec[iterationsCount * ensembleSize];
    final List<Func> weakModels = new ArrayList<>(iterationsCount * ensembleSize);
    final List<ScaledVectorFunc> ensemble = new ArrayList<>(iterationsCount * ensembleSize);
    final Mx cursor = new RowsVecArrayMx(new GradientCursor(learn, weakModels, B, probabilities, bds));

    for (int t = 0; t < iterationsCount; t++) {
      final Pair<Vec, Vec> factorize = this.factorize.factorize(cursor);

      // TODO: remove extra parameters
      for (int i = 0; i < ensembleSize; ++i) {
        B[t * ensembleSize + i] = factorize.second;
      }

      final L2 globalLoss = DataTools.newTarget(factory, factorize.first, learn);

      for (int i = 0; i < ensembleSize; ++i) {
        final ObliviousTree weakModel = (ObliviousTree) weak.fit(learn, DataTools.bootstrap(globalLoss, rfRnd));

        if (bds == null) {
          bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(weakModel.grid());
        }

        if (this.isGbdt) {
          IntStream.range(0, learn.length()).parallel().forEach(j -> {
            factorize.first.adjust(j, -weakModel.value(this.bds, j));
          });
        }

        weakModels.add(weakModel);
        ensemble.add(new ScaledVectorFunc(weakModel, factorize.second));
      }

      List<ScaledVectorFunc> ensembleUpdate = ensemble.subList(ensemble.size() - ensembleSize, ensemble.size());
      invoke(new Ensemble<>(ensembleUpdate, -step));
    }

    return new Ensemble<>(ensemble, -step);
  }

  private class GradientCursor extends Seq.Stub<Vec> {
    private final Mx cursor;
    private final VecDataSet learn;
    private final List<Func> weakModels;
    private final Mx target;
    private final Vec[] b;
    private final int[][] leafIndex;
    private final double[][][] buffer;

    private BinarizedDataSet bds;
    private int size = 0;

    public GradientCursor(VecDataSet learn, List<Func> weakModels, Vec[] b,
                          Mx target, BinarizedDataSet bds) {
      this.cursor = new VecBasedMx(learn.data().rows(), target.columns() - 1);
      this.learn = learn;
      this.weakModels = weakModels;
      this.target = target;
      this.b = b;
      this.bds = bds;
      this.leafIndex = new int[ensembleSize][learn.length()];
      this.buffer = new double[ensembleSize][][];
      initCursor();
    }

    private void initCursor() {
      for (int i = 0; i < learn.data().rows(); i++) {
        for (int j = 0; j < target.columns() - 1; j++) {
          cursor.adjust(i, j, 1.0 / target.columns() - target.get(i, j));
        }
      }
    }

    private void updateBuffer() {
      final int size = weakModels.size();
      final Vec b = this.b[size - 1];
      final double step = FMCBoosting.this.step;

      for (int tree = 0; tree < ensembleSize; ++tree) {
        ObliviousTree weakModel = (ObliviousTree) weakModels.get(size - ensembleSize + tree);
        List<BFGrid.Feature> features = weakModel.features();

        for (int index = 0; index < learn.length(); ++index) {
          int leaf = 0;
          for (int depth = 0; depth < features.size(); depth++) {
            leaf <<= 1;
            if (features.get(depth).value(bds.bins(features.get(depth).findex())[index])) {
              leaf++;
            }
          }
          leafIndex[tree][index] = leaf;
        }

        final double[] values = weakModel.values();

        if (buffer[tree] == null) {
          buffer[tree] = new double[values.length][target.columns() - 1];
        }

        for (int i = 0; i < values.length; ++i) {
          for (int j = 0; j < target.columns() - 1; ++j) {
            buffer[tree][i][j] = Math.exp(-step * b.get(j) * values[i]);
          }
        }
      }
    }

    private void updateCursor() {
      final int size = weakModels.size();
      final int classesCount = target.columns();

      if (bds == null) {
        bds = learn.cache().cache(Binarize.class, VecDataSet.class)
                .binarize(((ObliviousTree) weakModels.get(size - 1)).grid());
      }

      updateBuffer();

      IntStream.range(0, cursor.rows()).parallel().forEach(i -> {
        final Vec vec = cursor.row(i);

        double S = 1;
        for (int c = 0; c < classesCount - 1; c++) {
          double e = 1;
          for (int t = 0; t < ensembleSize; ++t) {
            e *= buffer[t][leafIndex[t][i]][c];
          }

          final double v = vec.get(c);
          final double value = target.get(i, c);
          S += (v + value) * (e - 1);
          vec.set(c, (v + value) * e);
        }

        for (int c = 0; c < classesCount - 1; c++) {
          vec.set(c, -target.get(i, c) + vec.get(c) / S);
        }
      });

      this.size = size;
    }

    @Override
    public Vec at(final int i) {
      if (weakModels.size() != size) {
        updateCursor();
      }
      return cursor.row(i);
    }

    @Override
    public Seq<Vec> sub(int start, int end) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Seq<Vec> sub(int[] indices) {
      throw new UnsupportedOperationException();
    }

    @Override
    public int length() {
      return target.rows();
    }

    @Override
    public boolean isImmutable() {
      return true;
    }

    @Override
    public Class<? extends Vec> elementType() {
      return Vec.class;
    }

    @SuppressWarnings("unchecked")
    @Override
    public Stream<Vec> stream() {
      return IntStream.range(0, length()).mapToObj(this::at);
    }
  }

  private static VecBasedMx getMultiLabelTarget(final Pool<?> pool) {
    // Convert multi-classification task to multi-label format
    if (pool.tcount() == 1) {
      final ArrayVec targets = (ArrayVec) pool.target(0);
      final int numClasses = (int) VecTools.max(targets) + 1;

      final VecBasedMx target = new VecBasedMx(pool.size(), numClasses);
      for (int i = 0; i < pool.size(); ++i) {
        target.set(i, (int) targets.get(i), 1.0);
      }

      return target;
    }

    // Multi-label task
    final VecBasedMx target = new VecBasedMx(pool.size(), pool.tcount());
    for (int j = 0; j < pool.tcount(); ++j) {
      VecTools.assign(target.col(j), (Vec) pool.target(j));
    }

    return target;
  }

  public static class Evaluator implements Consumer<Trans> {
    private static final int INTERVAL = 100;

    private final String name;
    private int iteration;
    private Metric[] metrics;

    private final Mx data;
    private final Mx target;
    private final Mx score;

    public Evaluator(final String name, final Pool<?> pool, final Metric... metrics) {
      this.name = name;
      this.iteration = 0;
      this.metrics = metrics;
      this.data = pool.vecData().data();
      this.target = FMCBoosting.getMultiLabelTarget(pool);
      this.score = new VecBasedMx(target.rows(), target.columns() - 1);
    }

    @Override
    public void accept(Trans trans) {
      Mx update = trans.transAll(data);
      MxTools.adjust(score, update);

      if (++iteration % INTERVAL == 0) {
        for (Metric metric : metrics) {
          final String metricName = metric.getClass().getSimpleName().toLowerCase();
          final Vec values = metric.apply(score, target);
          final String result = values.stream()
                  .mapToObj(it -> String.format("%.4f", it))
                  .collect(Collectors.joining(", "));
          System.out.println(String.format("%s %s: %s", name, metricName, result));
        }
      }
    }
  }


  public static abstract class Metric implements BiFunction<Mx, Mx, Vec> {
    public static class Accuracy extends Metric {
      @Override
      public Vec apply(Mx score, Mx target) {
        double matches = 0;

        for (int i = 0; i < score.rows(); ++i) {
          int clazz = VecTools.argmax(score.row(i));
          clazz = score.get(i, clazz) > 0 ? clazz : target.columns() - 1;
          int trueClazz = VecTools.argmax(target.row(i));
          matches += (clazz == trueClazz ? 1 : 0);
        }

        return new ArrayVec(matches / score.rows());
      }
    }

    public static class Precision extends Metric {
      public final int k;

      public Precision(int k) {
        this.k = k;
      }

      @Override
      public Vec apply(Mx score, Mx target) {
        double[] precision = new double[k];
        double[] values = new double[target.columns()];
        int[] indices = new int[target.columns()];

        for (int i = 0; i < score.rows(); ++i) {
          for (int j = 0; j < score.columns(); ++j) {
            values[j] = score.get(i, j);
            indices[j] = j;
          }

          // Add the last class
          values[score.columns()] = 0.0;
          indices[score.columns()] = score.columns();

          // TODO: use partition instead of sort
          ArrayTools.parallelSort(values, indices);

          for (int j = 0; j < k; ++j) {
            precision[j] += target.get(i, indices[indices.length - 1 - j]);
          }
        }

        for (int i = 1; i < k; ++i) {
          precision[i] += precision[i - 1];
        }

        for (int i = 0; i < k; ++i) {
          precision[i] = precision[i] / (i + 1) / score.rows();
        }

        return new ArrayVec(precision);
      }
    }
  }
}