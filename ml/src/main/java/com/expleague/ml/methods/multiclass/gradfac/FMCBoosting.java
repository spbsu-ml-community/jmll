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
import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.Pair;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.factorization.Factorization;
import com.expleague.ml.factorization.impl.StochasticALS;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.ScaledVectorFunc;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.ObliviousTree;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.lang.Math.exp;

/**
 * Experts League
 * Created by solar on 05.05.17.
 */
public class FMCBoosting extends WeakListenerHolderImpl<Trans> implements VecOptimization<BlockwiseMLLLogit> {
  protected final VecOptimization<L2> weak;
  private final Class<? extends L2> factory;
  private final Factorization factorize;
  private final int iterationsCount;
  private final double step;

  private BinarizedDataSet bds = null;

  public FMCBoosting(Factorization factorize, VecOptimization<L2> weak, int iterationsCount, double step) {
    this(factorize, weak, SatL2.class, iterationsCount, step);
  }

  public FMCBoosting(Factorization factorize, final VecOptimization<L2> weak, final Class<? extends L2> factory, final int iterationsCount, final double step) {
    this.factorize = factorize;
    this.weak = weak;
    this.factory = factory;
    this.iterationsCount = iterationsCount;
    this.step = step;
  }

  @Override
  public Ensemble<ScaledVectorFunc> fit(final VecDataSet learn, final BlockwiseMLLLogit globalLoss) {
    final Vec[] B = new Vec[iterationsCount];
    final List<Func> weakModels = new ArrayList<>(iterationsCount);
    final List<ScaledVectorFunc> ensamble = new ArrayList<>(iterationsCount);
    final Mx cursor = new RowsVecArrayMx(new LazyGradientCursor(learn, weakModels, B, globalLoss, bds));

    for (int t = 0; t < iterationsCount; t++) {
      if (t == 1) {
        initializeBinarizedDataSet(learn, (ObliviousTree) weakModels.get(0));
      }

      final Pair<Vec, Vec> factorize;
      if (this.factorize instanceof StochasticALS) {
        factorize = ((StochasticALS) this.factorize).factorize(cursor,
                t == 0 ? null : getLastWeakLearner(B[t - 1], (ObliviousTree) weakModels.get(t - 1), globalLoss));
      } else {
        factorize = this.factorize.factorize(cursor);
      }

      B[t] = factorize.second;
      final L2 localLoss = DataTools.newTarget(factory, factorize.first, learn);

      Interval.start();
      final Func weakModel = (Func) weak.fit(learn, localLoss);
      Interval.stopAndPrint("Fitting greedy oblivious tree");

      weakModels.add(weakModel);
      ensamble.add(new ScaledVectorFunc(weakModel, factorize.second));
      invoke(new Ensemble<>(ensamble, -step));
    }
    return new Ensemble<>(ensamble, -step);
  }

  private void initializeBinarizedDataSet(final VecDataSet learn, final ObliviousTree weakModel) {
    bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(weakModel.grid());
  }

  private BiConsumer<Integer, Vec> getLastWeakLearner(final Vec b, final ObliviousTree weakModel, BlockwiseMLLLogit target) {
    final int classesCount = target.classesCount();
    return (i, vec) -> {
      final int pointClass = target.label(i);
      final double scale = -step * weakModel.value(bds, i);

      double sum = 0;
      for (int c = 0; c < classesCount - 1; c++) {
        final double e = exp(b.get(c) * scale);
        final double v = vec.get(c);
        if (c == pointClass) {
          sum += (v + 1) * e;
          vec.set(c, (v + 1) * e);
        } else {
          sum += v * e;
          vec.set(c, v * e);
        }
      }

      for (int c = 0; c < classesCount - 1; c++) {
        if (c == pointClass) {
          vec.set(c, -1 + vec.get(c) / sum);
        } else {
          vec.set(c, vec.get(c) / sum);
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

    private BinarizedDataSet bds;
    private int size = 0;

    public GradientCursor(VecDataSet learn, List<Func> weakModels, Vec[] b, BlockwiseMLLLogit target, BinarizedDataSet bds) {
      this.cursor = new VecBasedMx(learn.data().rows(), target.classesCount() - 1);
      this.learn = learn;
      this.weakModels = weakModels;
      this.target = target;
      this.b = b;
      this.bds = bds;
    }

    public void updateCursor() {
      final int size = weakModels.size();

      IntStream.range(0, cursor.rows()).parallel().forEach(i -> {
        final ObliviousTree weakModel = (ObliviousTree) weakModels.get(size - 1);
        if (bds == null) {
          bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(weakModel.grid());
        }
        final BinarizedDataSet bds = this.bds;
        final double step = -FMCBoosting.this.step;
        final Vec b = this.b[size - 1];
        VecTools.incscale(cursor.row(i), b, weakModel.value(bds, i) * step);
      });

      this.size = size;
    }

    @Override
    public Vec at(final int i) {
      if (weakModels.size() != size) {
        updateCursor();
      }

      final int classesCount = target.classesCount();
      final Vec row = cursor.row(i);
      final Vec result = new ArrayVec(classesCount - 1);

      double sum = 0;
      for (int c = 0; c < classesCount - 1; c++) {
        final double expX = exp(row.get(c));
        sum += expX;
      }
      final int pointClass = target.label(i);
      for (int c = 0; c < classesCount - 1; c++) {
        if (pointClass == c)
          result.adjust(c, -1 + exp(row.get(c)) / (1. + sum));
        else
          result.adjust(c, exp(row.get(c)) / (1. + sum));
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

  private class LazyGradientCursor extends Seq.Stub<Vec> {
    private final VecDataSet learn;
    private final List<Func> weakModels;
    private final BlockwiseMLLLogit target;
    private final Vec[] b;
    private BinarizedDataSet bds;

    public LazyGradientCursor(VecDataSet learn, List<Func> weakModels, Vec[] b, BlockwiseMLLLogit target, BinarizedDataSet bds) {
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
        if (bds == null)
          bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(obliviousTree.grid());
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
        if (pointClass == c)
          result.adjust(c, -(1. + sum - exp(H_t.get(c))) / (1. + sum));
        else
          result.adjust(c, exp(H_t.get(c)) / (1. + sum));
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