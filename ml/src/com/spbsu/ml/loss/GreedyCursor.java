package com.spbsu.ml.loss;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.CursorOracle;

/**
 * User: solar
 * Date: 19.11.13
 * Time: 17:07
 */
public class GreedyCursor extends WeakListenerHolderImpl<CursorOracle.CursorMoved> implements CursorOracle<L2> {
  @Override
  public Vec cursor() {
    return null;  //To change body of implemented methods use File | Settings | File Templates.
  }

  @Override
  public Vec moveTo(Vec m) {
    return null;  //To change body of implemented methods use File | Settings | File Templates.
  }

  @Override
  public L2 local() {
    return null;  //To change body of implemented methods use File | Settings | File Templates.
  }

  @Override
  public double value(Vec x) {
    return 0;  //To change body of implemented methods use File | Settings | File Templates.
  }

  //package com.spbsu.ml.methods.trees;
//
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.math.vectors.VecTools;
//import com.spbsu.commons.math.vectors.impl.ArrayVec;
//import com.spbsu.commons.util.ArrayTools;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.Model;
//import com.spbsu.ml.Oracle1;
//import com.spbsu.ml.data.DataSet;
//import com.spbsu.ml.data.impl.BinarizedDataSet;
//import com.spbsu.ml.data.impl.Bootstrap;
//import com.spbsu.ml.loss.LLLogit;
//import com.spbsu.ml.methods.MLMethod;
//import com.spbsu.ml.models.ObliviousTree;
//
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Random;
//
///**
// * User: solar
// * Date: 30.11.12
// * Time: 17:01
// */
//public class GreedyObliviousClassificationTree implements MLMethod {
//  private final Random rng;
//  private final BinarizedDataSet ds;
//  private final int depth;
//  private BFGrid grid;
//
//  public GreedyObliviousClassificationTree(Random rng, DataSet ds, BFGrid grid, int depth) {
//    this.rng = rng;
//    this.depth = depth;
//    this.ds = new BinarizedDataSet(ds, grid);
//    this.grid = grid;
//  }
//
//  @Override
//  public Model fit(DataSet learn, Oracle1 loss) {
//    return fit(learn, loss, new ArrayVec(learn.power()));
//  }
//
//  @Override
//  public ObliviousTree fit(DataSet ds, Oracle1 loss, Vec point) {
//    if(!(loss instanceof LLLogit))
//      throw new IllegalArgumentException("Log likelihood with sigmoid value function supported only, found " + loss);
//    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
//    final Vec target = ds instanceof Bootstrap ? ((Bootstrap)ds).original().target() : ds.target();
//    LLClassificationLeaf seed = new LLClassificationLeaf(this.ds, point, target, VecTools.fill(new ArrayVec(point.dim()), 1.));
//    LLClassificationLeaf[] currentLayer = new LLClassificationLeaf[]{seed};
//    for (int level = 0; level < depth; level++) {
//      double[] scores = new double[grid.size()];
//      for (LLClassificationLeaf leaf : currentLayer) {
//        leaf.score(scores);
//      }
//      final int max = ArrayTools.max(scores);
//      if (max < 0)
//        throw new RuntimeException("Can not find optimal split!");
//      BFGrid.BinaryFeature bestSplit = grid.bf(max);
//      final LLClassificationLeaf[] nextLayer = new LLClassificationLeaf[1 << (level + 1)];
//      for (int l = 0; l < currentLayer.length; l++) {
//        LLClassificationLeaf leaf = currentLayer[l];
//        nextLayer[2 * l] = leaf;
//        nextLayer[2 * l + 1] = leaf.split(bestSplit);
//      }
//      conditions.add(bestSplit);
//      currentLayer = nextLayer;
//    }
//    final LLClassificationLeaf[] leaves = currentLayer;
//
//    double[] values = new double[leaves.length];
//    double[] weights = new double[leaves.length];
//    {
//      for (int i = 0; i < weights.length; i++) {
//        values[i] = leaves[i].alpha();
//        weights[i] = leaves[i].size();
//      }
//    }
//    return new ObliviousTree(conditions, values, weights);
//  }
//
//}
//package com.spbsu.ml.methods.trees;
//
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.math.vectors.VecTools;
//import com.spbsu.commons.math.vectors.impl.ArrayVec;
//import com.spbsu.commons.util.ArrayTools;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.Model;
//import com.spbsu.ml.MultiClassModel;
//import com.spbsu.ml.Oracle1;
//import com.spbsu.ml.data.DataSet;
//import com.spbsu.ml.data.DataTools;
//import com.spbsu.ml.data.impl.BinarizedDataSet;
//import com.spbsu.ml.data.impl.Bootstrap;
//import com.spbsu.ml.loss.LLLogit;
//import com.spbsu.ml.methods.MLMultiClassMethodOrder1;
//import com.spbsu.ml.models.ObliviousMultiClassTree;
//
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Random;
//
///**
// * User: solar
// * Date: 30.11.12
// * Time: 17:01
// */
//public class GreedyObliviousMultiClassificationTree implements MLMultiClassMethodOrder1 {
//    private final Random rng;
//    private final BinarizedDataSet ds;
//    private final int depth;
//    private BFGrid grid;
//
//    public GreedyObliviousMultiClassificationTree(Random rng, DataSet ds, BFGrid grid, int depth) {
//        this.rng = rng;
//        this.depth = depth;
//        this.ds = new BinarizedDataSet(ds, grid);
//        this.grid = grid;
//    }
//
//    @Override
//    public MultiClassModel fit(DataSet learn, Oracle1 loss) {
//        final Vec[] points = new Vec[DataTools.countClasses(learn.target())];
//        for (int i = 0; i < points.length; i++) {
//            points[i] = new ArrayVec(learn.power());
//        }
//        return fit(learn, loss, points);
//    }
//
//    @Override
//    public Model fit(DataSet learn, Oracle1 loss, Vec start) {
//        throw new UnsupportedOperationException("For multiclass methods continuation from fixed point is not supported");
//    }
//
//    public ObliviousMultiClassTree fit(DataSet ds, Oracle1 loss, Vec[] point) {
//        if (!(loss instanceof LLLogit))
//            throw new IllegalArgumentException("Log likelihood with sigmoid value function supported only");
//        final Vec target = ds instanceof Bootstrap ? ((Bootstrap) ds).original().target() : ds.target();
//        final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
//        MultiLLClassificationLeaf seed = new MultiLLClassificationLeaf(this.ds, point, target, VecTools.fill(new ArrayVec(ds.power()), 1.));
//        MultiLLClassificationLeaf[] currentLayer = new MultiLLClassificationLeaf[]{seed};
//        for (int level = 0; level < depth; level++) {
//            double[] scores = new double[grid.size()];
//            for (MultiLLClassificationLeaf leaf : currentLayer) {
//                leaf.score(scores);
//            }
//            final int max = ArrayTools.max(scores);
//            if (max < 0)
//                throw new RuntimeException("Can not find optimal split!");
//            BFGrid.BinaryFeature bestSplit = grid.bf(max);
//            final MultiLLClassificationLeaf[] nextLayer = new MultiLLClassificationLeaf[1 << (level + 1)];
//            for (int l = 0; l < currentLayer.length; l++) {
//                MultiLLClassificationLeaf leaf = currentLayer[l];
//                nextLayer[2 * l] = leaf;
//                nextLayer[2 * l + 1] = leaf.split(bestSplit);
//            }
//            conditions.add(bestSplit);
//            currentLayer = nextLayer;
//        }
//        final MultiLLClassificationLeaf[] leaves = currentLayer;
//
//        double[] values = new double[leaves.length];
//        double[] weights = new double[leaves.length];
//        boolean[][] masks = new boolean[leaves.length][];
//
//        {
//            for (int i = 0; i < weights.length; i++) {
//                values[i] = leaves[i].alpha();
//                masks[i] = leaves[i].mask();
//                weights[i] = leaves[i].size();
//            }
//        }
//        return new ObliviousMultiClassTree(conditions, values, weights, masks);
//    }
//}
//package com.spbsu.ml.methods.trees;
//
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.math.vectors.VecTools;
//import com.spbsu.commons.math.vectors.impl.ArrayVec;
//import com.spbsu.commons.util.ArrayTools;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.Model;
//import com.spbsu.ml.Oracle1;
//import com.spbsu.ml.data.DataSet;
//import com.spbsu.ml.data.impl.BinarizedDataSet;
//import com.spbsu.ml.data.impl.Bootstrap;
//import com.spbsu.ml.loss.CELogit;
//import com.spbsu.ml.methods.MLMethod;
//import com.spbsu.ml.models.ObliviousTree;
//
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
//import java.util.Random;
//
//import static java.lang.Math.*;
//
///**
// * User: solar
// * Date: 30.11.12
// * Time: 17:01
// */
//public class GreedyObliviousPACBayesClassificationTree implements MLMethod {
//  private final Random rng;
//  private final BinarizedDataSet ds;
//  private final int depth;
//  private BFGrid grid;
//  private Vec prevPoint;
//
//  public GreedyObliviousPACBayesClassificationTree(Random rng, DataSet ds, BFGrid grid, int depth) {
//    this.rng = rng;
//    this.depth = depth;
//    this.ds = new BinarizedDataSet(ds, grid);
//    this.grid = grid;
//    prevPoint = new ArrayVec(ds.power());
//  }
//
//  @Override
//  public Model fit(DataSet learn, Oracle1 loss) {
//    return fit(learn, loss, new ArrayVec(learn.power()));
//  }
//
//  @Override
//  public ObliviousTree fit(DataSet ds, Oracle1 loss, Vec point) {
//    if(!(loss instanceof CELogit))
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
//package com.spbsu.ml.methods.trees;
//
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.math.vectors.VecTools;
//import com.spbsu.commons.math.vectors.impl.ArrayVec;
//import com.spbsu.commons.util.ArrayTools;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.Model;
//import com.spbsu.ml.MultiClassModel;
//import com.spbsu.ml.Oracle1;
//import com.spbsu.ml.data.DataSet;
//import com.spbsu.ml.data.DataTools;
//import com.spbsu.ml.data.impl.BinarizedDataSet;
//import com.spbsu.ml.data.impl.Bootstrap;
//import com.spbsu.ml.loss.LLLogit;
//import com.spbsu.ml.methods.MLMultiClassMethodOrder1;
//import com.spbsu.ml.models.ObliviousMultiClassTree;
//
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Random;
//
///**
// * User: solar
// * Date: 30.11.12
// * Time: 17:01
// */
//public class GreedyObliviousSimpleMultiClassificationTree implements MLMultiClassMethodOrder1 {
//  private final Random rng;
//  private final BinarizedDataSet ds;
//  private final int depth;
//  private BFGrid grid;
//
//  public GreedyObliviousSimpleMultiClassificationTree(Random rng, DataSet ds, BFGrid grid, int depth) {
//    this.rng = rng;
//    this.depth = depth;
//    this.ds = new BinarizedDataSet(ds, grid);
//    this.grid = grid;
//  }
//
//  @Override
//  public MultiClassModel fit(DataSet learn, Oracle1 loss) {
//    final Vec[] points = new Vec[DataTools.countClasses(learn.target())];
//    for (int i = 0; i < points.length; i++) {
//      points[i] = new ArrayVec(learn.power());
//    }
//    return fit(learn, loss, points);
//  }
//
//  @Override
//  public Model fit(DataSet learn, Oracle1 loss, Vec start) {
//    throw new UnsupportedOperationException("For multiclass methods continuation from fixed point is not supported");
//  }
//
//  public ObliviousMultiClassTree fit(DataSet ds, Oracle1 loss, Vec[] point) {
//    if (!(loss instanceof LLLogit))
//      throw new IllegalArgumentException("Log likelihood with sigmoid value function supported only");
//    final Vec target = ds instanceof Bootstrap ? ((Bootstrap) ds).original().target() : ds.target();
//    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
//    MultiLLClassificationLeaf seed = new MultiLLClassificationLeaf(this.ds, point, target, VecTools.fill(new ArrayVec(ds.power()), 1.));
//    MultiLLClassificationLeaf[] currentLayer = new MultiLLClassificationLeaf[]{seed};
//    for (int level = 0; level < depth; level++) {
//      double[] scores = new double[grid.size()];
//      for (MultiLLClassificationLeaf leaf : currentLayer) {
//        leaf.score(scores);
//      }
//      final int max = ArrayTools.max(scores);
//      if (max < 0)
//        throw new RuntimeException("Can not find optimal split!");
//      BFGrid.BinaryFeature bestSplit = grid.bf(max);
//      final MultiLLClassificationLeaf[] nextLayer = new MultiLLClassificationLeaf[1 << (level + 1)];
//      for (int l = 0; l < currentLayer.length; l++) {
//        MultiLLClassificationLeaf leaf = currentLayer[l];
//        nextLayer[2 * l] = leaf;
//        nextLayer[2 * l + 1] = leaf.split(bestSplit);
//      }
//      conditions.add(bestSplit);
//      currentLayer = nextLayer;
//    }
//    final MultiLLClassificationLeaf[] leaves = currentLayer;
//
//    double[] values = new double[leaves.length];
//    double[] weights = new double[leaves.length];
//    boolean[][] masks = new boolean[leaves.length][];
//
//    {
//      for (int i = 0; i < weights.length; i++) {
//        values[i] = leaves[i].alpha();
//        masks[i] = leaves[i].mask();
//        weights[i] = leaves[i].size();
//      }
//    }
//    return new ObliviousMultiClassTree(conditions, values, weights, masks);
//  }
//}
//package com.spbsu.ml.methods.trees;
//
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.util.ArrayTools;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.data.impl.BinarizedDataSet;
//import com.spbsu.ml.data.impl.Bootstrap;
//
///**
//* User: solar
//* Date: 10.09.13
//* Time: 12:14
//*/
//public class LLClassificationLeaf implements BFLeaf {
//  private final Vec point;
//  private final Vec target;
//  private final Vec weight;
//  private final BinarizedDataSet ds;
//  private int[] indices;
//  protected final LLCounter[] counters;
//  LLCounter total = new LLCounter();
//  private final BFGrid.BFRow[] rows;
//
//  public LLClassificationLeaf(BinarizedDataSet ds, Vec point, Vec target, Vec weight) {
//    this(ds, point, target, weight, ds.original() instanceof Bootstrap ? ((Bootstrap) ds.original()).order() : ArrayTools.sequence(0, ds.original().power()));
//  }
//
//  public LLClassificationLeaf(BinarizedDataSet ds, int[] points, LLCounter right, LLClassificationLeaf bro) {
//    this.ds = ds;
//    point = bro.point;
//    target = bro.target;
//    weight = bro.weight;
//    this.indices = points;
//    total = right;
//    counters = new LLCounter[ds.grid().size() + ds.grid().rows()];
//    for (int i = 0; i < counters.length; i++) {
//      counters[i] = new LLCounter();
//    }
//    rows = ds.grid().allRows();
//  }
//
//  public LLClassificationLeaf(BinarizedDataSet ds, Vec point, Vec target, Vec weight, int[] pointIndices) {
//    this.ds = ds;
//    this.point = point;
//    this.indices = pointIndices;
//    this.target = target;
//    this.weight = weight;
//    for (int i = 0; i < this.indices.length; i++) {
//      final int index = indices[i];
//      total.found(this.point.get(index), target.get(index), weight.get(index));
//    }
//    counters = new LLCounter[ds.grid().size() + ds.grid().rows()];
//    for (int i = 0; i < counters.length; i++) {
//      counters[i] = new LLCounter();
//    }
//    rows = ds.grid().allRows();
//    ds.aggregate(this, this.indices);
//  }
//
//  @Override
//  public void append(int feature, byte bin, int index) {
//    counter(feature, bin).found(point.get(index), target.get(index), weight.get(index));
//  }
//
//  public LLCounter counter(int feature, byte bin) {
//    final BFGrid.BFRow row = rows[feature];
//    return counters[1 + feature + (bin > 0 ? row.bf(bin - 1).bfIndex : row.bfStart - 1)];
//  }
//
//  @Override
//  public int score(final double[] likelihoods) {
//    for (int f = 0; f < rows.length; f++) {
//      final BFGrid.BFRow row = rows[f];
//      LLCounter left = new LLCounter();
//      LLCounter right = new LLCounter();
//      right.add(total);
//
//      for (int b = 0; b < row.size(); b++) {
//        left.add(counter(f, (byte) b));
//        right.sub(counter(f, (byte) b));
//        likelihoods[row.bfStart + b] = left.score() + right.score();
//      }
//    }
//    return ArrayTools.max(likelihoods);
//  }
//
//  @Override
//  public int size() {
//    return indices.length;
//  }
//
//  /** Splits this leaf into two right side is returned */
//  @Override
//  public LLClassificationLeaf split(BFGrid.BinaryFeature feature) {
//    LLCounter left = new LLCounter();
//    LLCounter right = new LLCounter();
//    right.add(total);
//
//    for (int b = 0; b <= feature.binNo; b++) {
//      left.add(counter(feature.findex, (byte) b));
//      right.sub(counter(feature.findex, (byte) b));
//    }
//
//    final int[] leftPoints = new int[left.size()];
//    final int[] rightPoints = new int[right.size()];
//    final LLClassificationLeaf brother = new LLClassificationLeaf(ds, rightPoints, right, this);
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
//      ds.aggregate(brother, rightPoints);
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
//package com.spbsu.ml.methods.trees;
//
//import com.spbsu.commons.math.MathTools;
//
//import static java.lang.Math.*;
//
///** Key idea is to find \max_s \sum_i log \frac{1}{1 + e^{-(x_i + s})y_i}, where x_i -- current score, y_i \in \{-1,1\} -- category
// * for this we need to get solution for \sum_i \frac{y_i}{1 + e^{y_i(x_i + s}}. This equation is difficult to solve in closed form so
// * we use Taylor series approximation. For this we need to make substitution s = log(1-v) - log(1+v) so that Maclaurin series in terms of
// * v were limited.
// *
// * LLCounter is a class which calculates Maclaurin coefficients of original function and its first derivation.
// */
//public class LLCounter {
//  public volatile int good = 0;
//  public volatile int bad = 0;
//  public volatile double ll;
//  public volatile double lli;
//  public volatile double d1;
//  public volatile double d2;
//  public volatile double d3;
//  public volatile double d4;
//
//  public double alpha() {
//    double y = optimal();
////    final double n = min(good, bad);
////    final double m = max(good, bad);
//    return -log((1. - y) / (1. + y));// * (1 - log(2.)/log(n + 2.));
//  }
//  public double optimal() {
//    if (good == 0 || bad == 0)
//      return 0;
//    final double[] x = new double[3];
//
//    int cnt = MathTools.cubic(x, d4/6, d3/2, d2, d1);
//    double y = 0.;
//    double bestLL = 0;
//    for (int i = 0; i < cnt; i++) {
//      if (abs(x[i]) >= 0.8) // skip too optimistic solutions
//        continue;
//      final double score = scoreInner(x[i]);
//      if (score > bestLL) {
//        y = x[i];
//        bestLL = score;
//      }
//    }
//
//    return y;
//  }
//
//  private double scoreInner(double x) {
//    if (good == 0 || bad == 0)
//      return 0;
//    return d1 * x + d2 * x * x / 2 + d3 * x * x * x / 6 + d4 * x * x * x * x / 24;
//  }
//
//  public double fastScore() {
//    return ll - d1 * d1 / 2 / d2;
//  }
//
//  public double score() {
//    final double n = min(good, bad);
//    final double m = max(good, bad);
//    return score(0);//size() == 0 ? 0 : ll/size());//max(ll, lli)); // both masks in case of binary classification should be equally probable
//  }
//
//  public double score(double R ) {
//    final double optimal = optimal();
//    final double l = log((1. - optimal) / (1. + optimal));
//    R += -l*l/2. - 0.5 * log(2. * PI); // log N(0,1)
//    return scoreInner(optimal) + R; // in case of classification log prior must be _appended_ to ll
//  }
//
//  public void found(double current, double target, double weight) {
//    final double b = target > 0 ? 1. : -1.;
//    final double a = exp(-current*b);
//    final double d1 = -2 * a * b / (1 + a);
//    final double d2 = -d1 * d1 / a;
//    final double d3 = b * d2 * (a * a + 3) / (1 + a);
//    final double d4 = 12 * d2 * (1 + a * a * a) / (1 + a) / (1 + a);
//    ll += - weight * log(1 + a);
//    lli += - weight * log(1 + exp(current*b));
//    this.d1 += d1;
//    this.d2 += d2;
//    this.d3 += d3;
//    this.d4 += d4;
//    if (b > 0)
//      good++;
//    else
//      bad++;
//  }
//
//  public void add(LLCounter counter) {
//    ll += counter.ll;
//    lli += counter.lli;
//    d1 += counter.d1;
//    d2 += counter.d2;
//    d3 += counter.d3;
//    d4 += counter.d4;
//    good += counter.good;
//    bad += counter.bad;
//  }
//
//  public void sub(LLCounter counter) {
//    lli -= counter.lli;
//    ll -= counter.ll;
//    d1 -= counter.d1;
//    d2 -= counter.d2;
//    d3 -= counter.d3;
//    d4 -= counter.d4;
//    good -= counter.good;
//    bad -= counter.bad;
//  }
//
//  /**
//   * Combine two parts of $LL=-\sum_{\{a_c\}, b} \sum_c log(1+e^{-1^{I(b=c)}(-a-m(c)x)})$ depending on $m: C \to \{-1,1\}$
//   * actually they differs in sign of odd derivations so we need only to properly sum them :)
//   */
//  public void invert(LLCounter counter, double sign) {
//    d1 += sign * 2 * counter.d1;
//    d3 += sign * 2 * counter.d3;
//  }
//
//  public int size() {
//    return good + bad;
//  }
//}
//package com.spbsu.ml.methods.trees;
//
//import com.spbsu.commons.math.MathTools;
//
//import static java.lang.Math.*;
//
///** Key idea is to find minimal expectation of errors count: \min_s \sum_i \frac{1}{1 + e^{(x_i + s})y_i}, where x_i -- current score, y_i \in \{-1,1\} -- category
// * for this we need to get solution for \sum_i \frac{y_i}{1 + e^{y_i(x_i + s}}. This equation is difficult to solve in closed form so
// * we use Taylor series approximation. For this we need to make substitution s = log(1-v) - log(1+v) so that Maclaurin series in terms of
// * v were limited. Final optimization looks like:
// * \arg \min \sum_i \frac{1}{1 + \left(1 - t \over 1 + t\right)^y_i e^{y_i x_i}}
// * Maclaurin series (a = e^{x_i y_i}, b = y_i):
// * 1   \frac{1}{a + 1} +
// * x   \frac{2ab}}{(a + 1)^2} +
// * x^2 \frac{2ab^2(a - 1)}{(a + 1)^3} +
// * x^3 \frac{2ab(a^2(2b^2 + 1) + a(2 - 8b^2) + 2b^2 + 1)}{3(a + 1)^4} +
// * x^4 \frac{2ab^2(a^2(b^2 + 2) + a(4 - 10b^2) + b^2 + 2)}{3(a + 1)^5} +
// * MErrorCounter calculates Maclaurin coefficients.
// */
//public class MErrorsCounter {
//  public volatile int good = 0;
//  public volatile int bad = 0;
//  public volatile double m0;
//  public volatile double m1;
//  public volatile double m2;
//  public volatile double m3;
//  public volatile double m4;
//
//  public double alpha() {
//    double y = optimal();
//    return log((1. - y) / (1. + y));// * (1 - log(2.)/log(n + 2.));
//  }
//
//  public double optimal() {
//    if (good == 0 || bad == 0)
//      return 0;
//    final double[] x = new double[3];
//
//    int cnt = MathTools.cubic(x, 4 * m4, 3 * m3, 2 * m2, m1);
//    double y = 0.;
//    double bestLL = m0;
//    for (int i = 0; i < cnt; i++) {
//      final double normX = signum(x[i]) * min(abs(x[i]), 0.9);
//      final double score = scoreInner(normX);
//      if (score < bestLL) {
//        y = normX;
//        bestLL = score;
//      }
//    }
//
//    return y;
//  }
//
//  private double scoreInner(double x) {
//    return m0 + m1 * x + m2 * x * x + m3 * x * x * x + m4 * x * x * x * x;
//  }
//
//  public double score() {
//    return score(0);
//  }
//
//  public double score(double R) {
//    final double optimal = optimal();
////    final double l = log((1. - optimal) / (1. + optimal));
//    return scoreInner(optimal);// * R;
//  }
//
//  public void found(double current, double target, double weight) {
//    final double b = target > 0 ? 1. : -1.;
//    final double a = exp(current*b);
//    /*
//     * x   \frac{2ab}}{(a + 1)^2} +
//     * x^2 \frac{2ab^2(a - 1)}{(a + 1)^3} +
//     * x^3 \frac{2ab(a^2(2b^2 + 1) + a(2 - 8b^2) + 2b^2 + 1)}{3(a + 1)^4} +
//     * x^4 \frac{2ab^2(a^2(b^2 + 2) + a(4 - 10b^2) + b^2 + 2)}{3(a + 1)^5} +
//     * MErrorCounter calculates Maclaurin coefficients.
//     */
//    // \frac{1}{a + 1}
//    m0 += 1/(a + 1);
//    // \frac{2ab}}{(a + 1)^2}
//    final double m1 = 2 * a * b / (1 + a) / (1 + a);
//    // \frac{2ab^2(a - 1)}{(a + 1)^3}
//    final double m2 = 2 * a * b * b * (a - 1) / (1 + a) / (1 + a) / (1 + a);
//    // \frac{2ab(a^2(2b^2 + 1) + a(2 - 8b^2) + 2b^2 + 1)}{3(a + 1)^4}
//    final double m3 = 2 * a * b * (a * a * (2 * b * b + 1) + a * (2 - 8 * b * b) + 2 * b * b + 1) / 3 / (1 + a) / (1 + a) / (1 + a) / (1 + a);
//    // \frac{2ab^2(a^2(b^2 + 2) + a(4 - 10b^2) + b^2 + 2)}{3(a + 1)^5}
//    final double m4 = 2 * a * b * b * (a * a * (b * b + 2) + a * (4 - 10 * b * b) + b * b + 2) / 3 / (1 + a) / (1 + a) / (1 + a) / (1 + a) / (1 + a);
//    this.m1 += m1;
//    this.m2 += m2;
//    this.m3 += m3;
//    this.m4 += m4;
//    if (b > 0)
//      good++;
//    else
//      bad++;
//  }
//
//  public void add(MErrorsCounter counter) {
//    m0 += counter.m0;
//    m1 += counter.m1;
//    m2 += counter.m2;
//    m3 += counter.m3;
//    m4 += counter.m4;
//    good += counter.good;
//    bad += counter.bad;
//  }
//
//  public void sub(MErrorsCounter counter) {
//    m0 -= counter.m0;
//    m1 -= counter.m1;
//    m2 -= counter.m2;
//    m3 -= counter.m3;
//    m4 -= counter.m4;
//    good -= counter.good;
//    bad -= counter.bad;
//  }
//
//  /**
//   * Combine two parts of $LL=-\sum_{\{a_c\}, b} \sum_c log(1+e^{-1^{I(b=c)}(-a-m(c)x)})$ depending on $m: C \to \{-1,1\}$
//   * actually they differs in sign of odd derivations so we need only to properly sum them :)
//   */
//  public void invert(MErrorsCounter counter, double sign) {
//    m1 += sign * 2 * counter.m1;
//    m3 += sign * 2 * counter.m3;
//  }
//
//  public int size() {
//    return good + bad;
//  }
//}
//package com.spbsu.ml.methods.trees;
//
//import com.spbsu.commons.math.MathTools;
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.util.ArrayTools;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.data.impl.BinarizedDataSet;
//import com.spbsu.ml.data.impl.Bootstrap;
//import gnu.trove.TIntArrayList;
//
//import java.util.Arrays;
//
//import static java.lang.Math.exp;
//import static java.lang.Math.max;
//
///**
// * User: solar
// * Date: 10.09.13
// * Time: 12:14
// */
//public class MLLClassificationLeaf implements BFLeaf {
//  private final Vec[] point;
//  private final Vec target;
//  private final Vec weight;
//  private final BinarizedDataSet ds;
//  private final int[] indices;
//  private final double[] classWeights;
//  private final double[] classSums;
//  private final double[] classSums2;
//  private final double[] classWeightsTotal;
//  private final double[] classSumsTotal;
//  private final double[] classSums2Total;
//  private final BFGrid.BFRow[] rows;
//  private int classesCount;
//
//  public MLLClassificationLeaf(BinarizedDataSet ds, Vec[] point, Vec target, Vec weight) {
//    this(ds, point, target, weight, ds.original() instanceof Bootstrap ? ((Bootstrap) ds.original()).order() : ArrayTools.sequence(0, ds.original().power()));
//  }
//
//  public MLLClassificationLeaf(BinarizedDataSet ds, int[] points, MLLClassificationLeaf bro) {
//    this.ds = ds;
//    point = bro.point;
//    target = bro.target;
//    weight = bro.weight;
//    classesCount = bro.classesCount;
//    classWeights = new double[(ds.grid().size() + ds.grid().rows()) * classesCount];
//    classSums = new double[(ds.grid().size() + ds.grid().rows()) * classesCount];
//    classSums2 = new double[(ds.grid().size() + ds.grid().rows()) * classesCount];
//    classWeightsTotal = new double[classesCount];
//    classSumsTotal = new double[classesCount];
//    classSums2Total = new double[classesCount];
//
//    for (int i : points) {
//      final int classId = (int)target.get(points[i]);
//      classWeights[classId] += weight.get(points[i]);
//      classSums[classId] += weight.get(points[i]);
//      classSums2[classId] += weight.get(points[i]);
//      classWeightsTotal[classId] += weight.get(points[i]);
//      classWeights[classId] += weight.get(points[i]);
//    }
//    this.indices = points;
//    rows = ds.grid().allRows();
//  }
//
//  public MLLClassificationLeaf(BinarizedDataSet ds, Vec[] point, Vec target, Vec weight, int[] pointIndices) {
//    this.ds = ds;
//    this.point = point;
//    this.indices = pointIndices;
//    this.target = target;
//    this.classWeights = new double[ds.grid().size() * point.length];
//    this.classSums = new double[ds.grid().size() * point.length];
//    this.classSums2 = new double[ds.grid().size() * point.length];
//    this.weight = weight;
//    rows = ds.grid().allRows();
//    ds.aggregate(this, this.indices);
//  }
//
//  /**
//   *
//   */
//  @Override
//  public void append(int feature, byte bin, int index) {
//    final double w = weight.get(index);
//    int pointClass = (int)target.get(index);
//    final int tableOffset = (rows[feature].bfStart + feature + bin) * classesCount;
//    double sum = 0;
//    for (int c = 0; c < classesCount; c++) {
//      sum += exp(point[c].get(index));
//    }
//
//    for (int c = 0; c < classesCount; c++) {
//      final double ex = exp(point[c].get(index));
//
//      if (pointClass == c && c != 0) {
//        /*
//          positive and not last class examples give us:
//          m(c) {1 + \sum_{q\ne c} e^{s_q(x)} \over 1 + \sum_q e^{s_q(x)}}
//         */
//        final double v = (1. + sum - ex) / (1. + sum);
//        classWeights[tableOffset + c] += w;
//        classSums[tableOffset + c] += v;
//        classSums2[tableOffset + c] += v*v;
//      }
//      else {
//        /*
//          negative or last class examples give us:
//          - m(c) {e^{s_c(x)} \over 1 + \sum_q e^{s_q(x)}}
//         */
//        final double v = ex / (1. + sum);
//        classWeights[tableOffset + c] -= v;
//        classSums[tableOffset + c] -= v;
//        classSums2[tableOffset + c] += v*v ;
//      }
//    }
//  }
//
//  @Override
//  public int score(final double[] scores) {
//    int tableOffset = 0;
//    final double[] classSums = new double[classesCount];
//    final double[] classSums2 = new double[classesCount];
//    final double[] classWeights = new double[classesCount];
//    final double[] mask = new double[classesCount];
//    int findex = 0;
//    for (int f = 0; f < rows.length; f++) {
//      Arrays.fill(classSums, 0.);
//      Arrays.fill(classSums2, 0.);
//      Arrays.fill(classWeights, 0.);
//      for (int b = 0; b < rows[f].size(); b++) {
//        double sum = 0;
//        double weight = 0;
//
//        for (int c = 0; c < classesCount; c++) {
//          classSums[c] += this.classSums[tableOffset];
//          classSums2[c] += this.classSums2[tableOffset];
//          classWeights[c] += this.classWeights[tableOffset];
//          tableOffset++;
//
//          double sign = classSums[c] < MathTools.EPSILON ? (classSums[c] < -MathTools.EPSILON ? -1. : 0.) : 1.;
//          sum += sign * classSums[c];
//          weight += classWeights[c];
//        }
//        final double alpha = sum/weight;
//        double score = 0;
//        for (int c = 0; c < classesCount; c++) {
//          score += MathTools.sqr(max(0, classWeights[c] / (classWeights[c] - 1)))
//                   * (mask[c] * mask[c] * alpha * alpha - 2 * mask[c] * alpha * classSums[c]/ classWeights[c]);
//        }
//        scores[findex++] += score;
//      }
//      tableOffset += classesCount;
//    }
//    return ArrayTools.max(scores);
//  }
//
//  @Override
//  public int size() {
//    return indices.length;
//  }
//
//  /** Splits this leaf into two right side is returned */
//  @Override
//  public MLLClassificationLeaf split(BFGrid.BinaryFeature feature) {
//    final TIntArrayList left = new TIntArrayList(indices.length);
//    final TIntArrayList right = new TIntArrayList(indices.length);
//
//    final MLLClassificationLeaf brother = new MLLClassificationLeaf(ds, right.toNativeArray(), this);
//
//    int tableOffset = 0;
//    final double[] classSums = new double[classesCount];
//    final double[] classWeights = new double[classesCount];
//    final BFGrid.BFRow row = rows[feature.findex];
//    for (int b = 0; b < feature.binNo; b++) {
//      for (int c = 0; c < classesCount; c++) {
//        classSums[c] += this.classSums[tableOffset];
//        classSums2[c] += this.classSums2[tableOffset];
//        classWeights[c] += this.classWeights[tableOffset];
//        tableOffset++;
//      }
//    }
//    tableOffset += classesCount;
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
//      ds.aggregate(brother, rightPoints);
//    }
//    for (int i = 0; i < classSums.length; i++) {
//      classSums[i] -= brother.classSums[i];
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
//
//  public double m() {
//
//  }
//}
//package com.spbsu.ml.methods.trees;
//
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.math.vectors.VecTools;
//import com.spbsu.commons.math.vectors.impl.ArrayVec;
//import com.spbsu.commons.util.ArrayTools;
//import com.spbsu.commons.util.ThreadTools;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.data.DataTools;
//import com.spbsu.ml.data.impl.BinarizedDataSet;
//
//import java.util.concurrent.CountDownLatch;
//import java.util.concurrent.ThreadPoolExecutor;
//
//import static java.lang.Math.max;
//
///**
// * User: solar
// * Date: 10.09.13
// * Time: 12:23
// */
//public class MultiLLClassificationLeaf implements BFLeaf {
//  public static final double THRESHOLD = 3.;
//  private final BFGrid.BFRow[] rows;
//  private final LLClassificationLeaf[] folds;
//  private final int size;
//  private final int classesCount;
//  private final Vec[] point;
//  private final Vec target;
//  private final Vec weight;
//
//  public MultiLLClassificationLeaf(BinarizedDataSet ds, Vec[] point, Vec target, Vec weight) {
//    this.point = point;
//    this.target = target;
//    this.weight = weight;
//    classesCount = point.length;
//
//    folds = new LLClassificationLeaf[classesCount * classesCount];
//    Vec[] classTargets = new Vec[classesCount];
//    int[][] classIndices = new int[classesCount][];
//    for (int c = 0; c < classesCount; c++) {
//      classIndices[c] = DataTools.extractClass(target, c);
//      classTargets[c] = VecTools.fillIndices(VecTools.fill(new ArrayVec(target.dim()), -1.), classIndices[c], 1);
//    }
//
//    for (int c = 0; c < classesCount; c++) {
//      for (int t = 0; t < classesCount; t++) {
//        folds[c * classesCount + t] = new LLClassificationLeaf(ds, point[t], classTargets[t], weight, classIndices[c]);
//      }
//    }
//    rows = ds.grid().allRows();
//    size = ds.original().power();
//  }
//
//  private MultiLLClassificationLeaf(BFGrid.BFRow[] rows, LLClassificationLeaf[] folds, Vec[] point, Vec target, Vec weight) {
//    this.rows = rows;
//    this.folds = folds;
//    this.point = point;
//    this.target = target;
//    this.weight = weight;
//    classesCount = (int)Math.sqrt(folds.length);
//    int size = 0;
//    for (int i = 0; i < classesCount; i++) {
//      size += folds[i * classesCount + i].size();
//    }
//    this.size = size;
//  }
//
//  @Override
//  public void append(int feature, byte bin, int index) {
//    final int classId = (int) target.get(index);
//    for (int c = 0; c < classesCount; c++) { // iterate over instances of this point in LL for all classes
//      folds[classId * classesCount + c].append(feature, bin, index);
//    }
//  }
//
//  private static ThreadPoolExecutor executor;
//  private static synchronized ThreadPoolExecutor executor(int queueSize) {
//    if (executor == null || executor.getQueue().remainingCapacity() < queueSize) {
//      executor = ThreadTools.createBGExecutor("MultiLLClassificationLeaf optimizer", queueSize);
//    }
//    return executor;
//  }
//
//  @Override
//  public int score(double[] likelihoods) {
//    final Vec scores = VecTools.fill(new ArrayVec(likelihoods.length), Double.NEGATIVE_INFINITY);
//    final ThreadPoolExecutor exe = executor(rows.length);
//    final CountDownLatch latch = new CountDownLatch(rows.length);
//    for (int f = 0; f < rows.length; f++) {
//      final int finalF = f;
//      exe.execute(new Runnable() {
//        @Override
//        public void run() {
//          final BFGrid.BFRow row = rows[finalF];
//
//          LLCounter[] left = new LLCounter[classesCount];
//          LLCounter[] right = new LLCounter[classesCount];
//          LLCounter[] agg = new LLCounter[classesCount * row.size()];
//
//          {
//            for (int c = 0; c < classesCount; c++) {
//              left[c] = new LLCounter();
//              right[c] = new LLCounter();
//              for (int t = 0; t < right.length; t++) {
//                right[c].add(folds[t * classesCount + c].total);
//              }
//              for (int b = 0; b < row.size(); b++) {
//                agg[c * row.size() + b] = new LLCounter();
//                for (int t = 0; t < right.length; t++) {
//                  agg[c * row.size() + b].add(folds[t * classesCount + c].counter(finalF, (byte)b));
//                }
//              }
//            }
//          }
//
//          for (int b = 0; b < row.size(); b++) {
//            boolean[] mask = new boolean[classesCount];
//            for (int c = 0; c < classesCount; c++) {
//              left[c].add(agg[c * row.size() + b]);
//              right[c].sub(agg[c * row.size() + b]);
//            }
//            scores.set(row.bfStart + b, max(scores.get(row.bfStart + b),
//                    optimizeMask(left, mask).score() + optimizeMask(right, mask).score()));
//          }
//          latch.countDown();
//        }
//      });
//    }
//    try {
//      latch.await();
//    } catch (InterruptedException e) {
//      //
//    }
//    for (int i = 0; i < likelihoods.length; i++) {
//      likelihoods[i] += scores.get(i);
//    }
//    return ArrayTools.max(scores.toArray());
//  }
//
//  @Override
//  public MultiLLClassificationLeaf split(BFGrid.BinaryFeature feature) {
//    LLClassificationLeaf[] broLeaves = new LLClassificationLeaf[folds.length];
//    for (int i = 0; i < broLeaves.length; i++) {
//      broLeaves[i] = folds[i].split(feature);
//    }
//    return new MultiLLClassificationLeaf(rows, broLeaves, point, target, weight);
//  }
//
//  private static LLCounter optimizeMask(LLCounter[] folds, boolean[] mask) {
//    final int classesCount = mask.length;
//    final boolean[] confidence = new boolean[classesCount];
//    double denom = 0;
//    for (int c = 0; c < classesCount; c++) {
//      denom += folds[c].d2;
//    }
//    final LLCounter combined = new LLCounter();
//    double maxLL = 0;
//    for (int c = 0; c < classesCount; c++) {
//      combined.add(folds[c]);
//      maxLL = max(folds[c].ll, maxLL);
//    }
//    int bad = 0;
//    for (int c = 0; c < classesCount; c++) {
////      if (abs(folds[c].d1 * folds[c].d1/denom) > 10) {
////        confidence[c] = true;
////      }
//      mask[c] = folds[c].d1 > 0;
//      if (!mask[c]) {
//        bad += folds[c].good;
//        combined.invert(folds[c], -1);
//      }
//    }
//    if (combined.good < 2) {
////      System.out.println();
//      return combined;
//    }
//    combined.bad = bad;
//    combined.good -= combined.bad;
//    double bestScore = combined.score();
//    int best;
//    while(true) {
//      best = -1;
//      for (int c = 0; c < classesCount; c++) {
//        if (confidence[c]) // already in positive examples or definite negative
//          continue;
//        combined.invert(folds[c], mask[c] ? -1. : 1.);
//        final double score = combined.score();//maxLL);
//        if (bestScore < score) {
//          bestScore = score;
//          best = c;
//        }
//        combined.invert(folds[c], mask[c] ? 1. : -1.);
//      }
//      if (best >= 0) {
//        confidence[best] = true;
//        combined.invert(folds[best], mask[best] ? -1. : 1.);
//      }
//      else break;
//    }
////    boolean[] fullMask = new boolean[classesCount];
////    final double fullScore = optimizeMaskFull(folds, fullMask).score();
////    if (fullScore - combined.score() > 0.1) {
////      synchronized (System.class) {
////        for (int c = 0; c < classesCount; c++) {
////          System.out.println(abs(folds[c].d1 * folds[c].d1/denom) + " " + folds[c].d1 + " " + fullMask[c] + " " + mask[c] + " " + confidence[c]);
////        }
////        System.out.println();
////      }
////    }
//    return combined;
//  }
//
//  @Override
//  public int size() {
//    return size;
//  }
//
//  private boolean[] optimalMask;
//  @Override
//  public double alpha() {
//    this.optimalMask = new boolean[classesCount];
//    return optimize(this.optimalMask).alpha();
//  }
//
//  private LLCounter optimize(boolean[] mask) {
//    LLCounter[] classTotalsOpt = new LLCounter[classesCount];
//    for (int c = 0; c < classesCount; c++) {
//      classTotalsOpt[c] = new LLCounter();
//      for (int t = 0; t < classesCount; t++) {
//        classTotalsOpt[c].add(folds[t * classesCount + c].total);
//      }
//    }
//    return optimizeMask(classTotalsOpt, mask);
//  }
//
//  public boolean[] mask() {
//    if (optimalMask == null)
//      alpha();
//    return optimalMask;
//  }
//
//  public double score() {
//    return optimize(new boolean[classesCount]).score();
//  }
//}
//package com.spbsu.ml.methods.trees;
//
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.util.ArrayTools;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.data.impl.BinarizedDataSet;
//import com.spbsu.ml.data.impl.Bootstrap;
//
///**
//* User: solar
//* Date: 10.09.13
//* Time: 12:14
//*/
//public class PACBayesClassificationLeaf implements BFLeaf {
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
//    ds.aggregate(this, this.indices);
//  }
//
//  @Override
//  public void append(int feature, byte bin, int index) {
//    counter(feature, bin).found(point.get(index), target.get(index), weight.get(index));
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
//      ds.aggregate(brother, rightPoints);
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

}
