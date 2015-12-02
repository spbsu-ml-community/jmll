package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.linearRegressionExperiments.RidgeRegression;
import com.spbsu.ml.models.ModelTools;
import com.spbsu.ml.models.ObliviousTree;

import java.util.ArrayList;
import java.util.List;

/**
 * User: noxoomo
 * Date: 02.12.2015
 */

public class RidgeGreedyObliviousTree<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  private final GreedyObliviousTree<Loss> base;
  final double lambda;

  public RidgeGreedyObliviousTree(GreedyObliviousTree<Loss> base, double lambda) {
    this.base = base;
    this.lambda = lambda;
  }


  @Override
  public ModelTools.CompiledOTEnsemble  fit(final VecDataSet ds, final Loss loss) {
    ObliviousTree tree = base.fit(ds, loss);
    Ensemble<ObliviousTree> ensemble = new Ensemble<>(new ObliviousTree[]{tree}, VecTools.fill(new SingleValueVec(1), 1.0));
    ModelTools.CompiledOTEnsemble compiledOTEnsemble = ModelTools.compile(ensemble);
    List<ModelTools.CompiledOTEnsemble.Entry> entryList = compiledOTEnsemble.getEntries();

    Mx otData = new VecBasedMx(ds.data().rows(),entryList.size());
    final BinarizedDataSet bds =  ds.cache().cache(Binarize.class, VecDataSet.class).binarize(base.grid);
    final byte[] binary = new byte[base.grid.rows()];

    for (int i=0; i < otData.rows();++i) {
      for (int f=0; f <base.grid.rows();++f) {
        binary[f] = bds.bins(f)[i];
      }
      for (int j=0; j < otData.columns();++j) {
        final int[] bfIndices = entryList.get(j).getBfIndices();
        double increment = 1.0;
        for (int k = 0; k < bfIndices.length; k++) {
          if (!base.grid.bf(bfIndices[k]).value(binary)) {
            increment = 0;
            break;
          }
        }
        otData.set(i,j, increment);
      }
    }

    RidgeRegression ridgeRegression = new RidgeRegression(lambda);
    Vec weights = ridgeRegression.fit(otData, loss.target());
    ArrayList<ModelTools.CompiledOTEnsemble.Entry> newEntries  = new ArrayList<>(entryList);
    for (int i=0; i < weights.dim();++i) {
      newEntries.add(new ModelTools.CompiledOTEnsemble.Entry(entryList.get(i).getBfIndices(), weights.get(i)));
    }
    return new ModelTools.CompiledOTEnsemble(newEntries, tree.grid());
  }
}
