package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.trees.BFOptimizationSubset;
import com.expleague.ml.models.BumpyRegion;
import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;
import gnu.trove.list.array.TDoubleArrayList;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.List;

import static com.expleague.commons.math.MathTools.sqr;

/**
 * User: noxoomo
 */


public class GreedyTDBumpyRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;
  final double lambda;
  public GreedyTDBumpyRegion(final BFGrid grid, double lambda) {
    this.grid = grid;
    this.lambda = lambda;
  }


  class BasisRegression {
    final TDoubleArrayList means = new TDoubleArrayList();
    final TDoubleArrayList weights = new TDoubleArrayList();
    final TDoubleArrayList sums = new TDoubleArrayList();
    final TDoubleArrayList sd = new TDoubleArrayList();
    final ArrayList<TDoubleArrayList> correlations = new ArrayList<>();
    final TDoubleArrayList targetCorrelations = new TDoubleArrayList();
    final TDoubleArrayList prior = new TDoubleArrayList();
    final AdditiveStatistics targetStat;
    final double bias;
    final double targetSd;
    final Loss loss;

    public BasisRegression(Loss loss, AdditiveStatistics targetStat) {
      this.loss = loss;
      this.targetStat = targetStat;
      final double w = AdditiveStatisticsExtractors.weight(targetStat);
      final double sum = AdditiveStatisticsExtractors.sum(targetStat);
      final double sum2 = AdditiveStatisticsExtractors.sum2(targetStat);
      this.bias = sum / w;
      this.targetSd = FastMath.sqrt(sum2 / w - MathTools.sqr(sum / w));
    }


    double score(final double sum, final double weight) {
      final double factorBias = weight / AdditiveStatisticsExtractors.weight(targetStat);
      final double factorSd = FastMath.sqrt(factorBias * (1 - factorBias));

      if (weight < 5 || weight > (AdditiveStatisticsExtractors.weight(targetStat) - 5)) {
        return Double.POSITIVE_INFINITY;
      }

      final int m = means.size();
      final Mx cor = new VecBasedMx(m + 1, m + 1);
      final Vec targetCor = new ArrayVec(m + 1);

      for (int i = 0; i < m; ++i) {
        cor.set(i, i, 1.0 + prior.get(i));
        targetCor.set(i, targetCorrelations.get(i));
        for (int j = 0; j < i; ++j) {
          final double rho = correlations.get(i).get(j);
          cor.set(i, j, rho);
          cor.set(j, i, rho);
        }
      }

      cor.set(m, m, 1.0 + calcRegularization(weight));
      {
        double scale = 1.0 / (targetSd * factorSd) / AdditiveStatisticsExtractors.weight(targetStat);
        targetCor.set(m, (sum - factorBias * AdditiveStatisticsExtractors.sum(targetStat) - bias * weight + AdditiveStatisticsExtractors.weight(targetStat) * bias * factorBias) * scale);
      }

      for (int i = 0; i < m; ++i) {
        final double scale = 1.0 / sd.get(i) / factorSd / AdditiveStatisticsExtractors.weight(targetStat);
        final double fMean = means.get(i);
        final double fWeight = weights.get(i);
        final double rho = scale * (weight - fWeight * factorBias - weight * fMean + AdditiveStatisticsExtractors.weight(targetStat) * factorBias * fMean);
        cor.set(i, m, rho);
        cor.set(m, i, rho);
      }
      final Mx inv = MxTools.inverse(cor);
      Vec betas = new ArrayVec(m + 1);
      betas.set(0, bias * targetSd);

      Vec standardizedWeights = MxTools.multiply(inv, targetCor);
      for (int i = 0; i < m; ++i) {
        betas.adjust(0, -standardizedWeights.get(i) * means.get(i) * targetSd / sd.get(i));
        betas.set(i + 1, standardizedWeights.get(i) * targetSd / sd.get(i));
      }


      double c = 0;
      for (int i=0; i < betas.dim();++i) {
        c += betas.get(i);
      }

      double score = c * c * weight - 2 * c * sum;
      double w = weight;
      double s = sum;
      for (int i = sums.size(); i >0; --i) {
        c -= betas.get(i);
        w  = weights.get(i-1) - w;
        s = sums.get(i-1) - s;
        score += c * c * w - 2 * c * s;
        w = weights.get(i-1);
        s = sums.get(i-1);
      }

      return score;// * (1 + 2 * FastMath.log(weight + 1));// + Math.log(2) * sums.size();
    }

    void add(AdditiveStatistics inside) {
      final int m = means.size();
      final double factorSum = AdditiveStatisticsExtractors.sum(inside);
      final double factorWeight = AdditiveStatisticsExtractors.weight(inside);
      prior.add(calcRegularization(factorWeight));
      sums.add(AdditiveStatisticsExtractors.sum(inside));
      final double factorBias = factorWeight / AdditiveStatisticsExtractors.weight(targetStat);
      final double factorSd = FastMath.sqrt(factorBias * (1 - factorBias));
      means.add(factorBias);
      weights.add(factorWeight);
      sd.add(factorSd);

      {
        double scale = 1.0 / (targetSd * factorSd) / AdditiveStatisticsExtractors.weight(targetStat);
        targetCorrelations.add((factorSum - factorBias * AdditiveStatisticsExtractors.sum(targetStat) - bias * factorWeight + AdditiveStatisticsExtractors.weight(targetStat) * bias * factorBias) * scale);
      }

      TDoubleArrayList newCor = new TDoubleArrayList();
      for (int i = 0; i < m; ++i) {
        final double scale = 1.0 / sd.get(i) / factorSd / AdditiveStatisticsExtractors.weight(targetStat);
        final double fMean = means.get(i);
        final double fWeight = weights.get(i);
        final double rho = scale * (factorWeight - fWeight * factorBias - factorWeight * fMean + AdditiveStatisticsExtractors.weight(targetStat) * factorBias * fMean);
        newCor.add(rho);
      }
      correlations.add(newCor);
    }

    double calcRegularization(double weight) {
      final int k = correlations.size() + 1;
      double totalWeight = AdditiveStatisticsExtractors.weight(targetStat);
      double p = (weight  + 0.5) / (totalWeight + 1);
      double entropy = -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
      return lambda;// * Math.log(k);//   / entropy;
    }

    Vec estimateWeights() {
      int m = means.size();
      Mx cor = new VecBasedMx(m, m);
      Vec targetCor = new ArrayVec(m);
      for (int i = 0; i < m; ++i) {
        cor.set(i, i, 1.0 + prior.get(i));
        targetCor.set(i, targetCorrelations.get(i));
        for (int j = 0; j < i; ++j) {
          final double rho = correlations.get(i).get(j);
          cor.set(i, j, rho);
          cor.set(j, i, rho);
        }
      }
      Vec weights = new ArrayVec(m + 1);
      weights.set(0, bias * targetSd);

      if (m > 0) {
        Mx inv = MxTools.inverse(cor);
        Vec standardizedWeights = MxTools.multiply(inv, targetCor);
        for (int i = 0; i < m; ++i) {
          weights.adjust(0, -standardizedWeights.get(i) * means.get(i) * targetSd / sd.get(i));
          weights.set(i + 1, standardizedWeights.get(i) * targetSd / sd.get(i));
        }
      }
      return weights;
    }
  }

  @Override
  public BumpyRegion fit(final VecDataSet learn, final Loss loss) {
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(100);
    final boolean[] usedBF = new boolean[grid.size()];

    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    double currentScore = 1.0;
    final BFWeakConditionsOptimizationRegion current =
            new BFWeakConditionsOptimizationRegion(bds, loss, ((WeightedLoss) loss).points(), new BFGrid.BinaryFeature[0], new boolean[0], 0);
    final double[] scores = new double[grid.size()];
    final AdditiveStatistics[] stats = new AdditiveStatistics[grid.size()];

    BasisRegression estimator = new BasisRegression(loss, ((AdditiveStatistics) loss.statsFactory().create()).append(current.total()));

    while (conditions.size() < 6) {
      current.visitAllSplits((bf, left, right) -> {
        if (usedBF[bf.bfIndex]) {
          scores[bf.bfIndex] = Double.POSITIVE_INFINITY;
        } else {
          final AdditiveStatistics in = (AdditiveStatistics) loss.statsFactory().create();
          in.append(right);
          stats[bf.bfIndex] = in;
          scores[bf.bfIndex] = estimator.score(AdditiveStatisticsExtractors.sum(in), AdditiveStatisticsExtractors.weight(in));
        }
      });

      final int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || !Double.isFinite(scores[bestSplit]))
        break;


      if ((scores[bestSplit] + 1e-9 >= currentScore))
        break;

      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final BFOptimizationSubset outRegion = current.split(bestSplitBF, true);
      if (outRegion == null) {
        break;
      }

      conditions.add(bestSplitBF);
      usedBF[bestSplitBF.bfIndex] = true;
      currentScore = scores[bestSplit];
      estimator.add(stats[bestSplitBF.bfIndex]);
    }
    return new BumpyRegion(grid, conditions.toArray(new BFGrid.BinaryFeature[conditions.size()]), estimator.estimateWeights());
  }

}
