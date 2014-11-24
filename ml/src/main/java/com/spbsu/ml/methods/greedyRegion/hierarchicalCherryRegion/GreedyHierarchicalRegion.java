package com.spbsu.ml.methods.greedyRegion.hierarchicalCherryRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.CherryRegion;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/**
 * Created by noxoomo on 24/11/14.
 */
public class GreedyHierarchicalRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;


  public GreedyHierarchicalRegion (BFGrid grid) {
    this.grid = grid;
  }



  @Override
  public CherryRegion fit(final VecDataSet learn, final Loss loss) {
    final List<RegionLayer> layers = new ArrayList<>(10);
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    HierarchicalPick pick = new HierarchicalPick(bds,loss.statsFactory(),7,4.0);
    int[] points = ArrayTools.sequence(0, learn.length());

    double information = 0;
    double currentScore = Double.POSITIVE_INFINITY;
    while (true) {
      RegionLayer layer = pick.build(new Evaluator<AdditiveStatistics>() {
        @Override
        public double value(AdditiveStatistics stat) {
          return loss.score(stat);
        }
      }, points);
      if (currentScore * (1.0 / (1 +information + Math.log(layers.size()+1))) <=loss.score(layer.inside)* (1.0 / (1 + layer.information + information + Math.log(layers.size()+2))) + 1e-9) {// * (1 + 1.0 / (1  + Math.log(layers.size()+2))) + 1e-9) {
        break;
      }
      System.out.println("\n"  + layer.conditions);
      layers.add(layer);
      currentScore = loss.score(layer.inside);//layer.score;// * (1 + 1.0 / (1  + Math.log(layers.size()+1)));
      information += layer.information;
      points = layer.insidePoints;
    }
    BitSet[] conditions = new BitSet[layers.size()];
    for (int i=0; i < conditions.length;++i) {
      conditions[i] = layers.get(i).conditions;
    }

    CherryRegion region = new CherryRegion(conditions, 1, grid);
    Vec target = loss.target();
    double sum = 0;
    double weight = 0;
    for (int i = 0; i < bds.original().length(); ++i) {
      if (region.value(bds, i) == 1) {
        sum += target.get(i);
        ++weight;
      }
    }
    System.out.println("\nFound cherry region with " + weight + " points");
//    return new CherryRegion(conditions.toArray(new BitSet[conditions.size()]), weight > 0 ? sum / weight : 0, grid);
    return new CherryRegion(conditions, sum / weight,grid); //loss.bestIncrement(layers.get(layers.size()-1).inside), grid);
  }
}
