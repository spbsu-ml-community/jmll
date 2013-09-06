//package com.spbsu.ml.methods;
//
//import com.spbsu.commons.math.MathTools;
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.math.vectors.VecTools;
//import com.spbsu.commons.math.vectors.impl.ArrayVec;
//import com.spbsu.commons.util.ArrayTools;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.Model;
//import com.spbsu.ml.Oracle1;
//import com.spbsu.ml.data.Aggregator;
//import com.spbsu.ml.data.DataSet;
//import com.spbsu.ml.data.impl.BinarizedDataSet;
//import com.spbsu.ml.data.impl.Bootstrap;
//import com.spbsu.ml.loss.LogLikelihoodSigmoid;
//import com.spbsu.ml.models.ObliviousTree;
//import gnu.trove.TIntArrayList;
//
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Random;
//
//import static java.lang.Math.exp;
//import static java.lang.Math.log;
//
///**
// * User: solar
// * Date: 30.11.12
// * Time: 17:01
// */
//public class GreedyObliviousMultiClassificationTree extends GreedyObliviousClassificationTree {
//  private final Random rng;
//  private final BinarizedDataSet ds;
//  private final int depth;
//  private BFGrid grid;
//
//  public GreedyObliviousMultiClassificationTree(Random rng, DataSet ds, BFGrid grid, int depth) {
//    super(rng, ds, grid, depth);
//    this.rng = rng;
//    this.depth = depth;
//    this.ds = new BinarizedDataSet(ds, grid);
//    this.grid = grid;
//  }
//
//  @Override
//  public ObliviousTree fit(DataSet ds, Oracle1 loss, Vec point) {
//    if(!(loss instanceof LogLikelihoodSigmoid))
//      throw new IllegalArgumentException("Log likelihood with sigmoid probability function supported only");
//    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
//    final Vec target = ds instanceof Bootstrap ? ((Bootstrap)ds).original().target() : ds.target();
//    Leaf[] classes;
//
//    {
//      final int[] points = ds instanceof Bootstrap ? ((Bootstrap) ds).order() : ArrayTools.sequence(0, ds.power());
//      final List<TIntArrayList> classIndices = new ArrayList<TIntArrayList>();
//      for (int i = 0; i < points.length; i++) {
//        classIndices.get((int)target.get(points[i])).add(points[i]);
//      }
//      classes = new Leaf[classIndices.size()];
//      for (int i = 0; i < classes.length; i++) {
//        classes[i] = new Leaf(point, target, VecTools.fill(new ArrayVec(point.dim()), 1.), classIndices.get(i).toNativeArray());
//      }
//    }
//
//    for (int level = 0; level < depth; level++) {
//      double[] scores = new double[grid.size()];
//      for (Leaf leaf : leaves1) {
//        leaf.score(scores);
//      }
//      final int max = ArrayTools.max(scores);
//      if (max < 0)
//        throw new RuntimeException("Can not find optimal split!");
//      BFGrid.BinaryFeature bestSplit = grid.bf(max);
//      final Leaf[] nextLayer = new Leaf[1 << (level + 1)];
//      for (int i1 = 0; i1 < leaves1.length; i1++) {
//        Leaf leaf = leaves1[i1];
//        nextLayer[2 * i1] = leaf;
//        nextLayer[2 * i1 + 1] = leaf.split(bestSplit);
//      }
//      conditions.add(bestSplit);
//      leaves1 = nextLayer;
//    }
//    final Leaf[] leaves = leaves1;
//
//    double[] values = new double[leaves.length];
//    double[] weights = new double[leaves.length];
//    {
//      for (int i = 0; i < weights.length; i++) {
//        values[i] = leaves[i].alpha();
//        if (Double.isNaN(values[i]))
//          System.out.println(leaves[i].alpha());
//        weights[i] = leaves[i].points.length;
//      }
//    }
//    return new ObliviousTree(conditions, values, weights);
//  }
//}
