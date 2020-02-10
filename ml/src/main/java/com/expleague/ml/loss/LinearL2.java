package com.expleague.ml.loss;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.BFGrid;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.set.VecDataSet;

import java.util.Arrays;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

public class LinearL2 extends FuncC1.Stub implements AdditiveLoss<LinearL2.Stat>, TargetFunc {
  private final double lambda = 0.5;//1e-3;
  private final Mx subX;
  private final VecDataSet vds;
  private Vec target;
  private final int[] workingSet;

  public LinearL2(VecDataSet vds, BFGrid.Feature[] used, Vec target) {
    this.target = target;
    this.workingSet = Arrays.stream(used)/*.filter(bf -> bf.row().size() > 2)*/.mapToInt(BFGrid.Feature::findex).sorted().distinct().toArray();
    final Mx subX = new VecBasedMx(vds.length(), workingSet.length);
    for (int i = 0; i < vds.length(); i++) {
      final Vec row = vds.data().row(i);
      final int finalI = i;
      IntStream.range(0, workingSet.length).parallel().forEach(j -> subX.set(finalI, j, row.get(workingSet[j])));
    }
    this.subX = subX;
    this.vds = vds;
  }

  @Override
  public IntFunction<Stat> statsFactory() {
    return findex -> new Stat(findex, vds.data().col(findex));
  }

  @Override
  public int components() {
    return target.dim();
  }

  @Override
  public double value(int component, double x) {
    return MathTools.sqr(target.get(component) - x);
  }

  @Override
  public double value(Stat stats) {
    if (stats.weights < MathTools.EPSILON)
      return stats.sum2;
    Vec wHat = optimalWeightsRaw(stats);
    final Vec xy_covar = VecTools.copy(stats.xy.sub(0, wHat.dim()));
    VecTools.incscale(xy_covar, stats.sumX.sub(0, wHat.dim()), -stats.sum / stats.weights);
    VecTools.scale(xy_covar, 1. / stats.weights);
    return stats.sum2 - stats.sum * stats.sum / stats.weights - stats.weights * VecTools.multiply(wHat, xy_covar); // mD_\hat{w} - mD_0 = \|y_i\|^2 - \hat{w} Xy - \|y_i\|^2 = - \hat{w} Xy */
  }

  @Override
  public double score(Stat stats) {
    final double weight = stats.weights;
    if (weight < 2)
      return stats.sum2;
    Vec wHat = optimalWeightsRaw(stats);
    final Vec xy_covar = VecTools.copy(stats.xy.sub(0, wHat.dim()));
    VecTools.incscale(xy_covar, stats.sumX.sub(0, wHat.dim()), -stats.sum / weight);
    VecTools.scale(xy_covar, 1. / weight);
    final double reg = 1 + 0.005 * Math.log(weight + 1);
    final double scoreFromLinear = weight * VecTools.multiply(wHat, xy_covar);
    final double scoreFromConst = stats.sum * stats.sum / weight;
    final double targetValue = scoreFromConst + scoreFromLinear - lambda * VecTools.l2(wHat);
    return -targetValue * reg; // mD_\hat{w} - mD_0 = \|y_i\|^2 - \hat{w} Xy - \|y_i\|^2 = - \hat{w} Xy */
  }

  @Override
  public double bestIncrement(Stat comb) {
    return 0;
  }

  @Override
  public DataSet<?> owner() {
    return vds;
  }

  @Override
  public int dim() {
    return target.dim();
  }

  private Vec optimalWeightsRaw(Stat stats) {
    if (stats.weights < MathTools.EPSILON)
      return new ArrayVec(stats.xy.dim());
    int jIndex = Arrays.binarySearch(workingSet, stats.j);
    if (jIndex >= 0) { // reduce the dimensionality in case the feature is already in the working set
      final Mx inverseXXT = MxTools.inverseCholesky(stats.xxt.sub(0, 0, stats.xxt.rows() - 1, stats.xxt.columns() - 1));
      return MxTools.multiply(inverseXXT, stats.xy.sub(0, stats.xy.dim() - 1));
    }
    else {
      final Mx inverseXXT = MxTools.inverseCholesky(stats.xxt);
      return MxTools.multiply(inverseXXT, stats.xy);
    }
  }

  public Vec optimalWeights(Stat stats) {
    final Vec wHat = optimalWeightsRaw(stats);
    int jIndex = Arrays.binarySearch(workingSet, stats.j);
    if (-jIndex > workingSet.length || jIndex > 0)
      return wHat;
    jIndex = -jIndex; // plus 1 for bias
    final Vec wHatReordered = new ArrayVec(wHat.dim());
    for (int i = 0, index = 0; i < wHatReordered.dim(); i++, index++) {
      if (jIndex == i) {
        wHatReordered.set(i, wHat.get(wHat.dim() - 1)); // insert target feature
        index--; // shift source
      }
      else wHatReordered.set(i, wHat.get(index));
    }
    return wHatReordered;
  }

  public class Stat implements AdditiveStatistics {
    private final int j;
    private final Vec x_j;

    private double sum2;
    private double sum;
    private double weights;

    private Mx xxt;
    private Vec sumX;
    private Vec xy;

    public Stat(int j, Vec x_j) {
      this.j = j;
      this.x_j = x_j;
      int dim = 1 /* bias */ + subX.columns() + 1 /* j component */;
      this.xxt = new VecBasedMx(dim, dim);
      for (int i = 0; i < dim; i++) {
        this.xxt.set(i, i, lambda);
      }

      this.xy = new ArrayVec(dim);
      this.sumX = new ArrayVec(dim);
    }

    @Override
    public AdditiveStatistics append(int index, double weight) {
      final int dim = xxt.rows();
      final Vec x_i = subX.row(index);
      for (int u = 0; u < dim - 1; u++) {
        final double x_iu = u > 0 ? x_i.get(u - 1) : 1.;
        sumX.adjust(u, x_iu * weight);

        for (int v = 0; v < dim - 1; v++) {
          final double x_iv = v > 0 ? x_i.get(v - 1) : 1.;
          xxt.adjust(u, v, x_iu * x_iv * weight);
        }
      }
      // last dim of xxt and xy
      final double x_ji = x_j.get(index);
      final double y_i = target.get(index);
      for (int k = 0; k < dim - 1; k++) {
        final double x_ik = k > 0 ? x_i.get(k - 1) : 1;
        xxt.adjust(dim - 1, k, x_ji * x_ik * weight);
        xxt.adjust(k, dim - 1, x_ji * x_ik * weight);
        xy.adjust(k, y_i * x_ik * weight);
//        xxt.adjust(k, k, lambda * weight);
      }
      sumX.adjust(dim - 1, x_ji * weight);
      xxt.adjust(dim - 1, dim - 1, x_ji * x_ji * weight);
      xy.adjust(dim - 1, y_i * x_ji * weight);
      sum2 += y_i * y_i * weight;
      sum += y_i * weight;
      weights += weight;
      return this;
    }

    @Override
    public AdditiveStatistics append(AdditiveStatistics other) {
      if (j != ((Stat) other).j)
        throw new IllegalArgumentException();
      VecTools.append(xxt, ((Stat) other).xxt);
      VecTools.append(xy, ((Stat) other).xy);
      VecTools.append(sumX, ((Stat) other).sumX);
      sum2 += ((Stat) other).sum2;
      sum += ((Stat) other).sum;
      weights += ((Stat) other).weights;
      return this;
    }

    @Override
    public AdditiveStatistics remove(AdditiveStatistics other) {
      if (j != ((Stat) other).j)
        throw new IllegalArgumentException();
      VecTools.incscale(xxt, ((Stat) other).xxt, -1);
      VecTools.incscale(xy, ((Stat) other).xy, -1);
      VecTools.incscale(sumX, ((Stat) other).sumX, -1);
      sum2 -= ((Stat) other).sum2;
      sum -= ((Stat) other).sum;
      weights -= ((Stat) other).weights;
      return this;
    }

    @Override
    public AdditiveStatistics remove(int index, int times) {
      append(index, -times);
      return this;
    }

    @Override
    public AdditiveStatistics append(int index, int times) {
      return append(index, (double)times);
    }

    @Override
    public AdditiveStatistics remove(int index, double weight) {
      return append(index, -weight);
    }
  }
}
