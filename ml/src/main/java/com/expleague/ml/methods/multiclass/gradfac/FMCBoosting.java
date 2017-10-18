package com.expleague.ml.methods.multiclass.gradfac;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.factorization.Factorization;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.Pair;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.ScaledVectorFunc;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.ObliviousTree;

import java.util.ArrayList;
import java.util.List;

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
    final Mx cursor = new RowsVecArrayMx(new LazyGradientCursor(learn, weakModels, B, globalLoss));

    for (int t = 0; t < iterationsCount; t++) {
      final Pair<Vec, Vec> factorize = this.factorize.factorize(cursor);
      B[t] = factorize.second;
      final L2 localLoss = DataTools.newTarget(factory, factorize.first, learn);
      final Func weakModel = (Func) weak.fit(learn, localLoss);
      weakModels.add(weakModel);
      ensamble.add(new ScaledVectorFunc(weakModel, factorize.second));
      invoke(new Ensemble<>(ensamble, -step));
    }
    return new Ensemble<>(ensamble, -step);
  }

  private class LazyGradientCursor implements Seq<Vec> {
    private final VecDataSet learn;
    private final List<Func> weakModels;
    private final Vec[] b;
    private final BlockwiseMLLLogit target;
    private BinarizedDataSet bds;

    public LazyGradientCursor(VecDataSet learn, List<Func> weakModels, Vec[] b, BlockwiseMLLLogit target) {
      this.learn = learn;
      this.weakModels = weakModels;
      this.b = b;
      this.target = target;
    }

    @Override
    public Vec at(int i) {
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
      }
      else {
        final Vec vec = learn.at(i);
        for (int j = 0; j < size; j++) {
          VecTools.incscale(H_t, b[j], weakModels.get(j).value(vec) * step);
        }
      }
      final Vec result = new ArrayVec(classesCount - 1);
      double sum = 0;
      for (int c = 0; c < classesCount - 1; c++){
        final double expX = exp(H_t.get(c));
        sum += expX;
      }
      final int pointClass = target.label(i);
      for (int c = 0; c < classesCount - 1; c++){
        if (pointClass == c)
          result.adjust(c, -(1. + sum - exp(H_t.get(c)))/(1. + sum));
        else
          result.adjust(c, exp(H_t.get(c))/ (1. + sum));
      }
      return result;
    }

    @Override
    public Seq<Vec> sub(int start, int end) {
      throw new UnsupportedOperationException();
    }

    @Override
    public int length() {
      return learn.length();
    }

    @Override
    public boolean isImmutable() {
      return true;
    }

    @Override
    public Class<Vec> elementType() {
      return Vec.class;
    }
  }
}
