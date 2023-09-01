package com.expleague.otboost;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.*;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.*;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.FeatureSet;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.data.tools.PoolFSBuilder;
import com.expleague.ml.dynamicGrid.impl.BFDynamicGrid;
import com.expleague.ml.dynamicGrid.interfaces.DynamicGrid;
import com.expleague.ml.dynamicGrid.interfaces.DynamicRow;
import com.expleague.ml.dynamicGrid.trees.GreedyObliviousTreeDynamic;
import com.expleague.ml.dynamicGrid.trees.GreedyObliviousTreeDynamic2;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.*;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.impl.FeatureMetaImpl;
import com.expleague.ml.meta.impl.JsonDataSetMeta;
import com.expleague.ml.meta.impl.TargetMetaImpl;
import com.expleague.ml.meta.items.FakeItem;
import com.expleague.ml.methods.*;
import com.expleague.ml.methods.trees.GreedyObliviousLinearTree;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.ObliviousTree;

import java.util.Date;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import static com.expleague.commons.math.vectors.VecTools.copy;
import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: solar
 * Date: 26.11.12
 *
 * Time: 15:50
 */
public class ObliviousTreeBoostingTest extends GridTest {
  private FastRandom rng;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rng = new FastRandom(0);
  }

  public static class addBoostingListeners<GlobalLoss extends TargetFunc> {
    public addBoostingListeners(final GradientBoosting<GlobalLoss> boosting, final GlobalLoss loss, final Pool<?> _learn, final Pool<?> _validate) {
      final Consumer counter = new ProgressHandler() {
        int index = 0;
        @Override
        public void accept(final Trans partial) {
          System.out.print("\n" + index++ + "(" + Interval.time() + "ms)");
        }
      };
      final Consumer intervalFuse = ignore -> Interval.start();
      final ScoreCalcer learnListener = new ScoreCalcer(/*"\tlearn:\t"*/"\t", _learn.vecData(), _learn.target(L2.class), true, 1);
      final ScoreCalcer validateListener = new ScoreCalcer(/*"\ttest:\t"*/"\t", _validate.vecData(), _validate.target(L2.class), true, 1);
      final Consumer<Trans> modelPrinter = new ModelPrinter();
      boosting.addListener(counter);
      boosting.addListener(learnListener);
      boosting.addListener(validateListener);
      boosting.addListener(modelPrinter);
      boosting.addListener(intervalFuse);
      Interval.start();
      final Ensemble ans = boosting.fit(_learn.vecData(), loss);
      Vec current = new ArrayVec(_validate.size());
      for (int i = 0; i < _validate.size(); i++) {
        double f = 0;
        for (int j = 0; j < ans.size(); j++)
          f += ans.weight(j) * ((Func) ans.model(j)).value(_validate.vecData().data().row(i));
        current.set(i, f);
      }
      System.out.println("\n + Final loss = " + VecTools.distance(current, _validate.target(L2.class).target) / Math.sqrt(_validate.size()));
      Interval.start();
    }
  }

  public void testOTBoost() {
    final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
        new BootstrapOptimization<>(
            new GreedyObliviousTree<>(GridTools.medianGrid(learn.vecData(), 32), 6),
        rng),
        L2Reg.class, 2000, 0.005
    );
    new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
  }

  public void testDynamicOTBoost() {
    final GreedyObliviousTreeDynamic2<WeightedLoss<? extends L2>> gbdotDynamic = new GreedyObliviousTreeDynamic2<>(learn.vecData(), 6);
    final GradientBoosting<SatL2> boosting = new GradientBoosting<>(
            new BootstrapOptimization<>(gbdotDynamic, rng),
            L2Reg.class, 2000, 0.005
    );
    new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
    final DynamicGrid grid = gbdotDynamic.grid();
    for (int f = 0; f < grid.rows(); f++) {
      final DynamicRow row = grid.row(f);
      final StringBuilder rowRepresentation = new StringBuilder();
      rowRepresentation.append(f).append("(").append(row.size()).append(")");
      for (int b = 0; b < row.size(); b++) {
        rowRepresentation.append(" ").append(row.bf(b));
      }
      System.out.println(rowRepresentation);
    }
  }

  public void testOTLinearBoost() {
    final GradientBoosting<SatL2> boosting = new GradientBoosting<>(
        new BootstrapOptimization<>(
            new GreedyObliviousLinearTree<>(GridTools.medianGrid(learn.vecData(), 32), 6),
          rng),
        L2.class, 2000, 0.003
    );
    new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
  }

  public void testClassifyBoost() {
    final ProgressHandler pl = new ProgressHandler() {
      public static final int WSIZE = 10;
      final Vec cursor = new ArrayVec(learn.size());
      final double[] window = new double[WSIZE];
      int index = 0;
      double h_t = entropy(cursor);

      private double entropy(Vec cursor) {
        double result = 0;
        for (int i = 0; i < learn.target(0).length(); i++) {
          final double pX;
          if ((Double)learn.target(0).at(i) > 0)
            pX = 1./(1. + exp(-cursor.get(i)));
          else
            pX = 1./(1. + exp(cursor.get(i)));
          result += - pX * log(pX) / log(2);
        }
        return result;
      }

      @Override
      public void accept(Trans partial) {
        if (!(partial instanceof Ensemble))
          throw new RuntimeException("Can not work with other than ensembles");
        final Ensemble linear = (Ensemble) partial;
        final Trans increment = linear.last();
        Vec inc = copy(cursor);
        for (int i = 0; i < learn.size(); i++) {
          if (increment instanceof Ensemble) {
            cursor.adjust(i, linear.wlast() * (increment.trans(learn.vecData().data().row(i)).get(0)));
            inc.adjust(i, increment.trans(learn.vecData().data().row(i)).get(0));
          } else {
            cursor.adjust(i, linear.wlast() * ((Func) increment).value(learn.vecData().data().row(i)));
            inc.adjust(i, ((Func) increment).value(learn.vecData().data().row(i)));
          }
        }
        double h_t1 = entropy(inc);
        final double score = (h_t - h_t1) / info(increment);
        window[index % window.length] = score;
        index++;
        double meanScore = 0;
        for (int i = 0; i < window.length; i++)
          meanScore += Math.max(0, window[i]/ window.length);
        System.out.println("Info score: " + score + " window score: " + meanScore);
        h_t = entropy(cursor);
      }

      private double info(Trans increment) {
        if (increment instanceof ObliviousTree) {
          double info = 0;
          double total = 0;
          final double[] based = ((ObliviousTree) increment).based();
          for(int i = 0; i < based.length; i++) {
            final double fold = based[i];
            info += (fold + 1) * log(fold + 1);
            total += fold;
          }
          info -= log(total + based.length);
          info /= -(total + based.length);
          return info;
        }
        return Double.POSITIVE_INFINITY;
      }
    };

//    Action<Trans> learnScore = new ScorePrinter("Learn", learn.vecData(), learn.target(LLLogit.class));
//    Action<Trans> testScore = new ScorePrinter("Test", validate.vecData(), validate.target(LLLogit.class));
    Consumer<Trans> iteration = new Consumer<Trans>() {
      int index = 0;
      @Override
      public void accept(Trans trans) {
        System.out.println(index++);
      }
    };
    final GradientBoosting<LLLogit> boosting = new GradientBoosting<>(
        new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(learn.vecData(), 32), 6), rng),
        LOOL2.class, 2000, 0.05
    );
    boosting.addListener(iteration);
//    boosting.addListener(pl);
//    boosting.addListener(learnScore);
//    boosting.addListener(testScore);
    boosting.fit(learn.vecData(), learn.target(LLLogit.class));
  }

  public void testLinearModelVsObliviousTrees() {
    final FastRandom rng = new FastRandom(100500);
    final RandGaussianFeatureSet randfs = new RandGaussianFeatureSet(rng, 5);
    final JsonDataSetMeta meta = new JsonDataSetMeta("rng", "solar", new Date(), FakeItem.class, "noid");
    PoolFSBuilder<FakeItem> builder = new PoolFSBuilder<>(meta, FeatureSet.join(randfs, new LinearTarget(randfs, rng)));
    for (int i = 0; i < 100000; i++) {
      builder.accept(new FakeItem(i));
      builder.advance();
    }
    final Pool<FakeItem> itemPool = builder.create();
    final List<Pool<FakeItem>> split = DataTools.splitDataSet(itemPool, rng, 0.5, 0.5);
    final Pool<FakeItem> train = split.get(0);
    final Pool<FakeItem> validate = split.get(1);

    final GradientBoosting<L2> boosting = new GradientBoosting<>(
        new BootstrapOptimization<>(
            new GreedyObliviousTree<>(GridTools.medianGrid(train.vecData(), 32), 6),
            rng),
        L2Reg.class, 100000, 0.01
    );
    new addBoostingListeners<>(boosting, train.target(L2.class), train, validate);
  }

  public static void main(String[] args) {
//  public void testLinearModelVsLinearTrees() {
    final FastRandom rng = new FastRandom(100500);
    final RandGaussianFeatureSet randfs = new RandGaussianFeatureSet(rng, 10);
    final JsonDataSetMeta meta = new JsonDataSetMeta("rng", "solar", new Date(), FakeItem.class, "noid");
    final LinearTarget target = new LinearTarget(randfs, rng);
    PoolFSBuilder<FakeItem> builder = new PoolFSBuilder<>(meta, FeatureSet.join(randfs, target));
    for (int i = 0; i < 100000; i++) {
      builder.accept(new FakeItem(i));
      builder.advance();
    }
    final Pool<FakeItem> itemPool = builder.create();
    final List<Pool<FakeItem>> split = DataTools.splitDataSet(itemPool, rng, 0.5, 0.5);
    final Pool<FakeItem> train = split.get(0);
    final Pool<FakeItem> validate = split.get(1);
    System.out.println(target.w);
    final GradientBoosting<L2> boosting = new GradientBoosting<>(
        new BootstrapOptimization<>(
            new GreedyObliviousLinearTree<>(GridTools.medianGrid(train.vecData(), 32), 6),
            rng),
        L2Reg.class, 2000, 0.01
    );
    new addBoostingListeners<>(boosting, train.target(L2.class), train, validate);
  }

  public static class RandGaussianFeatureSet extends FeatureSet.Stub<FakeItem> {
    private final FastRandom rng;
    private final int dim;

    public RandGaussianFeatureSet(FastRandom rng, int dim) {
      super(IntStream.range(0, dim)
          .mapToObj(i -> new FeatureMetaImpl("rand-gaussian-" + i, "Random gaussian feature #" + i, FeatureMeta.ValueType.VEC))
          .toArray(FeatureMeta[]::new)
      );
      this.rng = rng;
      this.dim = dim;
      this.lastDraw = new ArrayVec(this.dim);
    }

    @Override
    public void accept(FakeItem item) {
      super.accept(item);
    }

    private final Vec lastDraw;
    @Override
    public Vec advanceTo(Vec to) {
      IntStream.range(0, dim).forEach(i -> {
        final double v = rng.nextGaussian();
        to.set(i, v);
        lastDraw.set(i, v);
      });
      return to;
    }

    public Vec lastDraw() {
      return lastDraw;
    }

    public int dim() {
      return dim;
    }
  }

  public static class LinearTarget extends FeatureSet.Stub<FakeItem> {
    private final Vec w;
    private final RandGaussianFeatureSet randfs;

    public LinearTarget(RandGaussianFeatureSet randfs, FastRandom rng) {
      super(new TargetMetaImpl("linear-target", "Synthetic linear target", FeatureMeta.ValueType.VEC));
      this.randfs = randfs;
      this.w = new ArrayVec(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
//      this.w = IntStream.range(0, randfs.dim()).mapToDouble(i -> rng.nextGaussian()).collect(VecBuilder::new, VecBuilder::append, VecBuilder::addAll).build();
    }

    @Override
    public Vec advanceTo(Vec to) {
      to.set(0, VecTools.multiply(w, randfs.lastDraw()));
      return to;
    }
  }
}


