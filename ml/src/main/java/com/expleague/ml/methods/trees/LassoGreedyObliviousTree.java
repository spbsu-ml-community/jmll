package com.expleague.ml.methods.trees;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.func.Linear;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.ElasticNetMethod;
import com.spbsu.commons.math.vectors.*;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.Binarize;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.ModelTools;
import com.expleague.ml.models.ObliviousTree;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * User: noxoomo
 * Date: 02.12.2015
 */

public class LassoGreedyObliviousTree<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  private final GreedyObliviousTree<Loss> base;
  final int nlambda;
  final double alpha;

  public LassoGreedyObliviousTree(GreedyObliviousTree<Loss> base, int nlambda, double alpha) {
    this.base = base;
    this.nlambda = nlambda;
    this.alpha = alpha;
  }

  private int[] learnPoints(Loss loss, VecDataSet ds) {
    if (loss instanceof WeightedLoss) {
      return ((WeightedLoss) loss).points();
    } else return ArrayTools.sequence(0, ds.length());
  }

  private int[] validationPoints(Loss loss) {
    if (loss instanceof WeightedLoss) {
      return ((WeightedLoss) loss).zeroPoints();
    } else {
      throw new RuntimeException("Wrong target type. No validation points");
    }
  }

  //TODO: noxoomo, remove duplicates
  //TODO: noxoomo, no intercept regularization…
  @SuppressWarnings("Duplicates")
  private Pair<Mx, Vec> filter(final List<ModelTools.CompiledOTEnsemble.Entry> entryList, final BinarizedDataSet bds, Vec sourceTarget, int[] points) {
    final byte[] binary = new byte[base.grid.rows()];
    Mx otData = new VecBasedMx(points.length, entryList.size());
    Vec target = new ArrayVec(points.length);


    for (int i = 0; i < points.length; ++i) {
      final int ind = points[i];
      for (int f = 0; f < base.grid.rows(); ++f) {
        binary[f] = bds.bins(f)[ind];
      }

      for (int j = 0; j < otData.columns(); ++j) {
        final int[] bfIndices = entryList.get(j).getBfIndices();
        double increment = 1.0;
        for (int k = 0; k < bfIndices.length; k++) {
          if (!base.grid.bf(bfIndices[k]).value(binary)) {
            increment = 0;
            break;
          }
        }
        otData.set(i, j, increment);
        target.set(i, sourceTarget.get(ind));
      }
    }
    return new Pair<>(otData, target);
  }

  @Override
  public ModelTools.CompiledOTEnsemble fit(final VecDataSet ds, final Loss loss) {
    ObliviousTree tree = base.fit(ds, loss);
    Ensemble<ObliviousTree> ensemble = new Ensemble<>(new ObliviousTree[]{tree}, VecTools.fill(new SingleValueVec(1), 1.0));
    ModelTools.CompiledOTEnsemble compiledOTEnsemble = ModelTools.compile(ensemble);
    List<ModelTools.CompiledOTEnsemble.Entry> entryList = compiledOTEnsemble.getEntries().stream()
            .filter(entry -> entry.getBfIndices().length > 0 && entry.getValue() != 0).collect(Collectors.toList());

    Vec target = VecTools.copy(loss.target());
    double bias = 0;
    for (int i = 0; i < target.dim(); ++i) {
      bias += target.get(i);
    }
    bias /= target.dim();
    for (int i = 0; i < target.dim(); ++i) {
      target.adjust(i, -bias);
    }


    final BinarizedDataSet bds = ds.cache().cache(Binarize.class, VecDataSet.class).binarize(base.grid);

    Pair<Mx, Vec> compiledLearn = filter(entryList, bds, target, learnPoints(loss, ds));

    Vec entryBias = colMean(compiledLearn.first);
    center(compiledLearn.first, entryBias);


    Pair<Mx, Vec> compiledValidate = filter(entryList, bds, target, validationPoints(loss));
    center(compiledValidate.first, entryBias);

    ElasticNetMethod lasso = new ElasticNetMethod(1e-4, alpha, 1.0);
    List<Linear> weightsPath = lasso.fit(compiledLearn.first, compiledLearn.second, nlambda);
    double[] scores = weightsPath.parallelStream().mapToDouble(linear -> {
      final double testL2 = VecTools.distanceL2(linear.transAll(compiledValidate.first), compiledValidate.second) / compiledValidate.first.rows();
      final double learnL2 = VecTools.distanceL2(linear.transAll(compiledLearn.first), compiledLearn.second) / compiledLearn.first.rows();
      return 0.63 * testL2 + (1-0.63) * learnL2;
    }).toArray();
    int best = 0;
    double bestScore = scores[0];
    for (int i = 0; i < scores.length; ++i) {
      if (scores[i] < bestScore) {
        bestScore = scores[i];
        best = i;
      }
    }
    Vec weights = weightsPath.get(best).weights;
    ArrayList<ModelTools.CompiledOTEnsemble.Entry> newEntries = new ArrayList<>();
    for (int i = 0; i < weights.dim(); ++i) {
      if (weights.get(i) != 0) {
        newEntries.add(new ModelTools.CompiledOTEnsemble.Entry(entryList.get(i).getBfIndices(), weights.get(i)));
        bias -= weights.get(i) * entryBias.get(i);
      }
    }
    if (bias != 0) {
      newEntries.add(new ModelTools.CompiledOTEnsemble.Entry(new int[0], bias));
    }

    System.out.println("Next entries batch: " + newEntries.size() + " of " + (entryList.size() + 1) + " nonZero");
    return new ModelTools.CompiledOTEnsemble(newEntries, tree.grid());
  }

  private Vec colMean(Mx data) {
    Vec result = new ArrayVec(data.columns());
    for (int i = 0; i < data.rows(); ++i) {
      for (int j = 0; j < data.columns(); ++j) {
        result.adjust(j, data.get(i, j));
      }
    }
    return VecTools.scale(result, 1.0 / data.rows());
  }

  private void center(Mx data, Vec bias) {
    for (int j = 0; j < data.rows(); ++j) {
      for (int i = 0; i < data.columns(); ++i) {
        data.adjust(i, j, -bias.get(i));
      }
    }
  }
}
