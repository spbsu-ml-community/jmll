//package com.spbsu.ml.methods.trees;
//
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
//import com.spbsu.commons.random.FastRandom;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.data.impl.BinarizedDataSet;
//import com.spbsu.ml.data.set.VecDataSet;
//import com.spbsu.ml.loss.LLLogit;
//import com.spbsu.ml.loss.StatBasedLoss;
//import com.spbsu.ml.methods.VecOptimization;
//import com.spbsu.ml.models.ObliviousTree;
//import com.sun.jdi.Bootstrap;
//
//import java.util.ArrayList;
//import java.util.List;
//
///**
//* Created by noxoomo on 06/10/14.
//*/
//public class GreedyObliviousPACBayesClassificationTree<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
//  private final FastRandom random;
//  private final int depth;
//  private BFGrid grid;
//
//  public GreedyObliviousPACBayesClassificationTree(FastRandom random, BFGrid grid, int depth) {
//    this.random = random;
//    this.depth = depth;
//    this.grid = grid;
//  }
//
//
//  @Override
//  public ObliviousTree fit(VecDataSet ds, final Loss loss) {
//    if(!(loss instanceof LLLogit))
//      throw new IllegalArgumentException("Log likelihood with sigmoid value function supported only, found " + loss);
//    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
//    final Vec target = ds instanceof Bootstrap ? ((Bootstrap)ds).original().target() : ds.target();
//    PACBayesClassificationLeaf seed = new PACBayesClassificationLeaf(this.ds, point, target, VecTools.fill(new ArrayVec(point.dim()), 1.));
//    PACBayesClassificationLeaf[] currentLayer = new PACBayesClassificationLeaf[]{seed};
//    PACBayesClassificationLeaf seedPrev = new PACBayesClassificationLeaf(this.ds, prevPoint, target, VecTools.fill(new ArrayVec(point.dim()), 1.));
//    PACBayesClassificationLeaf[] currentLayerPrev = new PACBayesClassificationLeaf[]{seedPrev};
//    for (int level = 0; level < depth; level++) {
//      double[] scores = new double[grid.size()];
//      double[] s = new double[grid.size()];
//      double[] sPrev = new double[grid.size()];
//      for (int i = 0; i < currentLayer.length; i++) {
//        Arrays.fill(s, 0);
//        Arrays.fill(sPrev, 0);
//        currentLayer[i].score(s);
//        currentLayerPrev[i].score(sPrev);
//        double sumPrev = 0;
//        double minPrev = sPrev[ArrayTools.min(sPrev)];
//        double maxPrev = sPrev[ArrayTools.max(sPrev)];
//        for (int j = 0; j < sPrev.length; j++) {
//          final double normalized = -(sPrev[j] - minPrev)*3/(maxPrev - minPrev);
//          sumPrev += exp(normalized * normalized);
//        }
//
//        for (int j = 0; j < sPrev.length; j++) {
//          if (currentLayer[i].size() > 0) {
//            final double normalized = -(sPrev[j] - minPrev)*3/(maxPrev - minPrev + 0.1);
//            final double p = exp(-normalized * normalized);
//            final double R = sqrt((-log(p) + log(1 / 0.05)) / 2. / currentLayer[i].size());
//            if(Double.isNaN(R))
//              System.out.println();
//            scores[j] += s[j] + (Double.isNaN(R) ? 0 : R);
//          }
//          else
//            scores[j] += s[j];
//        }
//      }
//
//      final int min = ArrayTools.min(scores);
//      if (min < 0)
//        throw new RuntimeException("Can not find optimal split!");
//      BFGrid.BinaryFeature bestSplit = grid.bf(min);
//      final PACBayesClassificationLeaf[] nextLayer = new PACBayesClassificationLeaf[1 << (level + 1)];
//      final PACBayesClassificationLeaf[] nextLayerPrev = new PACBayesClassificationLeaf[1 << (level + 1)];
//      double errors = 0;
//      double realErrors = 0;
//      for (int l = 0; l < currentLayer.length; l++) {
//        PACBayesClassificationLeaf leaf = currentLayer[l];
//        PACBayesClassificationLeaf leafPrev = currentLayerPrev[l];
//        nextLayer[2 * l] = leaf;
//        nextLayer[2 * l + 1] = leaf.split(bestSplit);
//        nextLayerPrev[2 * l] = leafPrev;
//        nextLayerPrev[2 * l + 1] = leafPrev.split(bestSplit);
//        errors += nextLayer[2 * l].total.score();
//        errors += nextLayer[2 * l + 1].total.score();
//        realErrors += checkLeaf(point, target, nextLayer[2 * l]);
//        realErrors += checkLeaf(point, target, nextLayer[2 * l + 1] );
//      }
////      System.out.println("Estimated error: " + realErrors + " found: " + errors);
//      conditions.add(bestSplit);
//      currentLayer = nextLayer;
//      currentLayerPrev = nextLayerPrev;
//    }
//    final PACBayesClassificationLeaf[] leaves = currentLayer;
//
//    double[] values = new double[leaves.length];
//    double[] weights = new double[leaves.length];
//    {
//      for (int i = 0; i < weights.length; i++) {
//        values[i] = leaves[i].alpha();
//        weights[i] = leaves[i].size();
//      }
//    }
//    VecTools.assign(prevPoint, point);
//    return new ObliviousTree(conditions, values, weights);
//  }
//
//  private double checkLeaf(Vec point, Vec target, PACBayesClassificationLeaf pacBayesClassificationLeaf) {
//    double inc = pacBayesClassificationLeaf.total.alpha();
//    double xx = 0;
//    for (int i = 0; i < pacBayesClassificationLeaf.indices.length; i++) {
//      int index = pacBayesClassificationLeaf.indices[i];
//      final double y = target.get(index) > 0 ? 1 : -1;
//      xx += 1./(1 + Math.exp(y * (point.get(index) + inc)));
//    }
//    return xx;
//  }
//}
//
//
//class PACBayesClassificationLeaf implements BFLeaf {
//  private final Vec point;
//  private final Vec target;
//  private final Vec weight;
//  private final BinarizedDataSet ds;
//  int[] indices;
//  protected final MErrorsCounter[] counters;
//  MErrorsCounter total = new MErrorsCounter();
//  private final BFGrid.BFRow[] rows;
//
//  public PACBayesClassificationLeaf(BinarizedDataSet ds, Vec point, Vec target, Vec weight) {
//    this(ds, point, target, weight, ds.original() instanceof Bootstrap ? ((Bootstrap) ds.original()).order() : ArrayTools.sequence(0, ds.original().power()));
//  }
//
//  public PACBayesClassificationLeaf(BinarizedDataSet ds, int[] points, MErrorsCounter right, PACBayesClassificationLeaf bro) {
//    this.ds = ds;
//    point = bro.point;
//    target = bro.target;
//    weight = bro.weight;
//    this.indices = points;
//    total = right;
//    counters = new MErrorsCounter[ds.grid().size() + ds.grid().rows()];
//    for (int i = 0; i < counters.length; i++) {
//      counters[i] = new MErrorsCounter();
//    }
//    rows = ds.grid().allRows();
//  }
//
//  public PACBayesClassificationLeaf(BinarizedDataSet ds, Vec point, Vec target, Vec weight, int[] pointIndices) {
//    this.ds = ds;
//    this.point = point;
//    this.indices = pointIndices;
//    this.target = target;
//    this.weight = weight;
//    for (int i = 0; i < this.indices.length; i++) {
//      final int index = indices[i];
//      total.found(this.point.get(index), target.get(index), weight.get(index));
//    }
//    counters = new MErrorsCounter[ds.grid().size() + ds.grid().rows()];
//    for (int i = 0; i < counters.length; i++) {
//      counters[i] = new MErrorsCounter();
//    }
//    rows = ds.grid().allRows();
//    ds.aggregate(this, target, this.point, this.indices);
//  }
//
//  @Override
//  public void append(int feature, byte bin, double target, double current, double weight) {
//    counter(feature, bin).found(current, target, weight);
//  }
//
//  public MErrorsCounter counter(int feature, byte bin) {
//    final BFGrid.BFRow row = rows[feature];
//    return counters[1 + feature + (bin > 0 ? row.bf(bin - 1).bfIndex : row.bfStart - 1)];
//  }
//
//  @Override
//  public int score(final double[] likelihoods) {
//    for (int f = 0; f < rows.length; f++) {
//      final BFGrid.BFRow row = rows[f];
//      MErrorsCounter left = new MErrorsCounter();
//      MErrorsCounter right = new MErrorsCounter();
//      right.add(total);
//
//      for (int b = 0; b < row.size(); b++) {
//        left.add(counter(f, (byte) b));
//        right.sub(counter(f, (byte) b));
//        likelihoods[row.bfStart + b] += left.score() + right.score();
//      }
//    }
//    return ArrayTools.min(likelihoods);
//  }
//
//  @Override
//  public int size() {
//    return indices.length;
//  }
//
//  /** Splits this leaf into two right side is returned */
//  @Override
//  public PACBayesClassificationLeaf split(BFGrid.BinaryFeature feature) {
//    MErrorsCounter left = new MErrorsCounter();
//    MErrorsCounter right = new MErrorsCounter();
//    right.add(total);
//
//    for (int b = 0; b <= feature.binNo; b++) {
//      left.add(counter(feature.findex, (byte) b));
//      right.sub(counter(feature.findex, (byte) b));
//    }
//
//    final int[] leftPoints = new int[left.size()];
//    final int[] rightPoints = new int[right.size()];
//    final PACBayesClassificationLeaf brother = new PACBayesClassificationLeaf(ds, rightPoints, right, this);
//
//    {
//      int leftIndex = 0;
//      int rightIndex = 0;
//      byte[] bins = ds.bins(feature.findex);
//      byte splitBin = (byte)feature.binNo;
//      for (int i = 0; i < indices.length; i++) {
//        final int point = indices[i];
//        if (bins[point] > splitBin)
//          rightPoints[rightIndex++] = point;
//        else
//          leftPoints[leftIndex++] = point;
//      }
//      ds.aggregate(brother, target, point, rightPoints);
//    }
//    for (int i = 0; i < counters.length; i++) {
//      counters[i].sub(brother.counters[i]);
//    }
//    indices = leftPoints;
//    total = left;
//    return brother;
//  }
//
//  @Override
//  public double alpha() {
//    return total.alpha();
//  }
//}
