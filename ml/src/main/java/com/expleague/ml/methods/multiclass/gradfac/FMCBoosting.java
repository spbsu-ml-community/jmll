package com.expleague.ml.methods.multiclass.gradfac;

import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.factorization.Factorization;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.ScaledVectorFunc;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.loss.multiclass.MCMacroF1Score;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.ObliviousTree;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.lang.Math.exp;

/**
 * Experts League Created by solar on 05.05.17.
 */
public class FMCBoosting extends WeakListenerHolderImpl<Trans> implements
    VecOptimization<BlockwiseMLLLogit> {

  protected final VecOptimization<StatBasedLoss> weak;
  private final Class<? extends L2> factory;
  private final Factorization factorize;
  private final int iterationsCount;
  private double step;
  private final boolean lazyCursor;
  private final int ensembleSize;
  private final boolean isGbdt;
  private BinarizedDataSet bds = null;
  private final FastRandom rfRnd = new FastRandom(13);

  private VecDataSet valid;
  private BlockwiseMLLLogit validTarget;
  private int bestIterCount = 0;
  private double bestAccuracy;
  private int earlyStoppingRounds = 0;

  public FMCBoosting(final Factorization factorize, final VecOptimization<StatBasedLoss> weak,
      final int iterationsCount, final double step, final boolean lazyCursor) {
    this(factorize, weak, SatL2.class, iterationsCount, step, lazyCursor, 1, false);
  }

  public FMCBoosting(final Factorization factorize, final VecOptimization<StatBasedLoss> weak,
      final int iterationsCount, final double step) {
    this(factorize, weak, SatL2.class, iterationsCount, step, false, 1, false);
  }

  public FMCBoosting(final Factorization factorize, final VecOptimization<StatBasedLoss> weak,
      final int iterationsCount, final double step, final int ensembleSize) {
    this(factorize, weak, SatL2.class, iterationsCount, step, false, ensembleSize, false);
  }

  public FMCBoosting(final Factorization factorize, final VecOptimization<StatBasedLoss> weak,
      final Class<? extends L2> factory, final int iterationsCount, final double step,
      final int ensembleSize, final boolean isGbdt) {
    this(factorize, weak, factory, iterationsCount, step, false, ensembleSize, isGbdt);
  }

  public FMCBoosting(final Factorization factorize, final VecOptimization<StatBasedLoss> weak,
      final Class<? extends L2> factory, final int iterationsCount, final double step) {
    this(factorize, weak, factory, iterationsCount, step, false, 1, false);
  }

  public FMCBoosting(Factorization factorize, final VecOptimization<StatBasedLoss> weak,
      final Class<? extends L2> factory, final int iterationsCount, final double step,
      final boolean lazyCursor, final int ensembleSize, final boolean isGbdt) {
    this.factorize = factorize;
    this.weak = weak;
    this.factory = factory;
    this.iterationsCount = iterationsCount;
    this.step = step;
    this.lazyCursor = lazyCursor;
    this.ensembleSize = ensembleSize;
    this.isGbdt = isGbdt;
  }

  @Override
  public Ensemble<ScaledVectorFunc> fit(final VecDataSet learn, final BlockwiseMLLLogit target) {
    final boolean silent = valid == null;

    final double[] validScores = new double[iterationsCount];
    final Vec[] B = new Vec[iterationsCount * ensembleSize];
    final List<Func> weakModels = new ArrayList<>(iterationsCount * ensembleSize);
    final List<ScaledVectorFunc> ensamble = new ArrayList<>(iterationsCount * ensembleSize);
    final Mx cursor;
    if (lazyCursor) {
      cursor = new RowsVecArrayMx(new LazyGradientCursor(learn, weakModels, B, target, bds));
    } else {
      cursor = new RowsVecArrayMx(new GradientCursor(learn, weakModels, B, target, bds));
    }

    Vec validClass = null;
    final MCMacroF1Score f1MacroScore =
        valid != null ? new MCMacroF1Score(validTarget.labels(), valid) : null;

    VecBasedMx validScore = null;
    if (valid != null) {
      validScore = new VecBasedMx(valid.length(), target.classesCount() - 1);
    }

    for (int t = 0; t < iterationsCount; t++) {
      if ((t + 1) % 20 == 0) {
        System.out.println("Iteration " + (t + 1));
      }

      final Pair<Vec, Vec> factorize = this.factorize.factorize(cursor);

      // TODO: remove extra parameters
      for (int i = 0; i < ensembleSize; ++i) {
        B[t * ensembleSize + i] = factorize.second;
      }

      final L2 globalLoss = DataTools.newTarget(factory, factorize.first, learn);

      if (!silent) {
        Interval.start();
      }

      for (int i = 0; i < ensembleSize; ++i) {
        final ObliviousTree weakModel = (ObliviousTree) weak
            .fit(learn, DataTools.bootstrap(globalLoss, rfRnd));

        if (bds == null) {
          bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(weakModel.grid());
        }

        if (this.isGbdt) {
          IntStream.range(0, learn.length()).parallel().forEach(j -> {
            factorize.first.adjust(j, -weakModel.value(this.bds, j));
          });
        }

        weakModels.add(weakModel);
        ensamble.add(new ScaledVectorFunc(weakModel, factorize.second));
      }

      // Update valid score
      if (valid != null) {
        Interval.start();

        if (validClass == null) {
          validClass = new ArrayVec(valid.length());
        }

        for (int i = 0; i < ensembleSize; ++i) {
          final int index = ensamble.size() - ensembleSize + i;
          final ScaledVectorFunc func = ensamble.get(index);
          for (int j = 0; j < valid.length(); ++j) {
            VecTools.append(validScore.row(j), VecTools.scale(func.trans(valid.at(j)), -step));
          }
        }

        if ((t + 1) % 20 == 0) {
          double matches = 0;
          for (int i = 0; i < valid.length(); ++i) {
            double[] score = validScore.row(i).toArray();
            int clazz = ArrayTools.max(score);
            clazz = score[clazz] > 0 ? clazz : target.classesCount() - 1;
            matches += (clazz == validTarget.label(i) ? 1 : 0);

            // save class for i-th sample
            validClass.set(i, clazz);
          }

          final double accuracy = matches / valid.length();
          final double f1Score = f1MacroScore.value(validClass);

          validScores[t] = f1Score;

          if (bestIterCount == 0 || accuracy > bestAccuracy) {
            bestIterCount = t + 1;
            bestAccuracy = accuracy;
          }
          System.out.println(String.format("Valid accuracy: %.4f", accuracy));
          System.out.println(String.format("Valid f1 macro: %.4f", f1Score));
        }

        if (earlyStoppingRounds > 0 && t + 1 - bestIterCount == earlyStoppingRounds) {
          System.out.println("Early stopping!");
          break;
        }
      }

      invoke(new Ensemble<>(ensamble, -step));
    }

    if (valid != null) {
      try {
        final String result = DoubleStream.of(validScores).mapToObj(Double::toString)
            .collect(Collectors.joining(","));
        final PrintStream out = new PrintStream(new FileOutputStream(new File("valid_scores.txt")));
        out.println(result);
        out.close();
      } catch (Exception e) {
        // pass
      }

      System.out.println(String.format(String.format("Best iterations count: %d", bestIterCount)));
      System.out.println(String.format(String.format("Best valid accuracy: %.4f", bestAccuracy)));
      return new Ensemble<>(ensamble.subList(0, ensembleSize * bestIterCount), -step);
    }

    return new Ensemble<>(ensamble, -step);
  }

  public void setEarlyStopping(final VecDataSet valid, final BlockwiseMLLLogit validTarget,
      final int earlyStoppingRounds) {
    this.valid = valid;
    this.validTarget = validTarget;
    this.earlyStoppingRounds = earlyStoppingRounds;
  }

  private BiConsumer<Integer, Vec> getLastWeakLearner(final Vec b, final ObliviousTree weakModel,
      BlockwiseMLLLogit target) {
    final int classesCount = target.classesCount();
    return (i, vec) -> {
      final int pointClass = target.label(i);
      final double scale = -step * weakModel.value(bds, i);

      double S = 1;
      for (int c = 0; c < classesCount - 1; c++) {
        final double e = exp(b.get(c) * scale);
        final double v = vec.get(c);
        if (c == pointClass) {
          S += (v + 1) * (e - 1);
          vec.set(c, (v + 1) * e);
        } else {
          S += v * (e - 1);
          vec.set(c, v * e);
        }
      }

      for (int c = 0; c < classesCount - 1; c++) {
        if (c == pointClass) {
          vec.set(c, -1 + vec.get(c) / S);
        } else {
          vec.set(c, vec.get(c) / S);
        }
      }
    };
  }

  private class GradientCursor extends Seq.Stub<Vec> {

    private final Mx cursor;
    private final VecDataSet learn;
    private final List<Func> weakModels;
    private final BlockwiseMLLLogit target;
    private final Vec[] b;
    private final int[][] leafIndex;
    private final double[][][] buffer;

    private BinarizedDataSet bds;
    private int size = 0;

    public GradientCursor(VecDataSet learn, List<Func> weakModels, Vec[] b,
        BlockwiseMLLLogit target, BinarizedDataSet bds) {
      this.cursor = new VecBasedMx(learn.data().rows(), target.classesCount() - 1);
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
        for (int j = 0; j < target.classesCount() - 1; j++) {
          cursor.adjust(i, j, 1.0 / target.classesCount());
          if (j == target.label(i)) {
            cursor.adjust(i, j, -1);
          }
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
          buffer[tree] = new double[values.length][target.classesCount() - 1];
        }

        for (int i = 0; i < values.length; ++i) {
          for (int j = 0; j < target.classesCount() - 1; ++j) {
            buffer[tree][i][j] = Math.exp(-step * b.get(j) * values[i]);
          }
        }
      }
    }

    private void updateCursor() {
      final int size = weakModels.size();
      final int classesCount = target.classesCount();

      if (bds == null) {
        bds = learn.cache().cache(Binarize.class, VecDataSet.class)
            .binarize(((ObliviousTree) weakModels.get(size - 1)).grid());
      }

      // long timeStart = System.currentTimeMillis();
      updateBuffer();
      // System.out.println("updateBuffer: " + (System.currentTimeMillis() - timeStart) + " (ms)");

      // timeStart = System.currentTimeMillis();
      IntStream.range(0, cursor.rows()).parallel().forEach(i -> {
        final Vec vec = cursor.row(i);
        final int pointClass = target.label(i);

        double S = 1;
        for (int c = 0; c < classesCount - 1; c++) {
          double e = 1;
          for (int t = 0; t < ensembleSize; ++t) {
            e *= buffer[t][leafIndex[t][i]][c];
          }

          final double v = vec.get(c);
          if (c == pointClass) {
            S += (v + 1) * (e - 1);
            vec.set(c, (v + 1) * e);
          } else {
            S += v * (e - 1);
            vec.set(c, v * e);
          }
        }

        for (int c = 0; c < classesCount - 1; c++) {
          if (c == pointClass) {
            vec.set(c, -1 + vec.get(c) / S);
          } else {
            vec.set(c, vec.get(c) / S);
          }
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
      return target.dim() / target.blockSize();
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

  private class LazyGradientCursor extends Seq.Stub<Vec> {

    private final VecDataSet learn;
    private final List<Func> weakModels;
    private final BlockwiseMLLLogit target;
    private final Vec[] b;
    private BinarizedDataSet bds;

    public LazyGradientCursor(VecDataSet learn, List<Func> weakModels, Vec[] b,
        BlockwiseMLLLogit target, BinarizedDataSet bds) {
      this.learn = learn;
      this.weakModels = weakModels;
      this.target = target;
      this.b = b;
      this.bds = bds;
    }

    @Override
    public Vec at(final int i) {
      final int classesCount = target.classesCount();
      final Vec H_t = new ArrayVec(classesCount - 1);

      final List<Func> weakModels = this.weakModels;
      final int size = weakModels.size();
      final double step = -FMCBoosting.this.step;
      if (size > 0 && weakModels.get(0) instanceof ObliviousTree) {
        final ObliviousTree obliviousTree = (ObliviousTree) weakModels.get(0);
        if (bds == null) {
          bds = learn.cache().cache(Binarize.class, VecDataSet.class)
              .binarize(obliviousTree.grid());
        }
        final BinarizedDataSet bds = this.bds;
        for (int j = 0; j < size; j++) {
          final ObliviousTree tree = (ObliviousTree) weakModels.get(j);
          VecTools.incscale(H_t, b[j], tree.value(bds, i) * step);
        }
      } else {
        final Vec vec = learn.at(i);
        for (int j = 0; j < size; j++) {
          VecTools.incscale(H_t, b[j], weakModels.get(j).value(vec) * step);
        }
      }
      final Vec result = new ArrayVec(classesCount - 1);
      double sum = 0;
      for (int c = 0; c < classesCount - 1; c++) {
        final double expX = exp(H_t.get(c));
        sum += expX;
      }
      final int pointClass = target.label(i);
      for (int c = 0; c < classesCount - 1; c++) {
        if (pointClass == c) {
          result.adjust(c, -(1. + sum - exp(H_t.get(c))) / (1. + sum));
        } else {
          result.adjust(c, exp(H_t.get(c)) / (1. + sum));
        }
      }
      return result;
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
      return target.dim() / target.blockSize();
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
}