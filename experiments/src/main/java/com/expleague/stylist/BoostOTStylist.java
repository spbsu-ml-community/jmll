package com.expleague.stylist;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeqBuilder;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.BFGrid;
import com.expleague.ml.GridTools;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.cli.builders.data.impl.DataBuilderCrossValidation;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.loss.LOOL2;
import com.expleague.ml.loss.PLogit;
import com.expleague.ml.loss.RLogit;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.TargetMeta;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.ModelTools;
import com.expleague.ml.models.ObliviousTree;

import java.io.IOException;
import java.util.List;
import java.util.function.Consumer;

import static com.expleague.ml.cli.builders.data.ReaderFactory.createFeatureTxtReader;

public class BoostOTStylist {
  public static void main(String[] args) throws IOException {
    final FastRandom rng = new FastRandom(0);
    final DataBuilderCrossValidation cvBuilder = new DataBuilderCrossValidation();
    cvBuilder.setReader(createFeatureTxtReader());
    cvBuilder.setLearnPath(System.getenv("HOME") + "/data/pools/green/stylist-20140702.txt");
    final Pool<?> pool = DataTools.loadFromFeaturesTxt(System.getenv("HOME") + "/data/pools/green/stylist-20140702.txt");
    final IntSeqBuilder classifyTarget = new IntSeqBuilder();
    final Vec target = (Vec)pool.target(0);
    for (int i = 0; i < target.length(); i++) {
      if (target.get(i) > 2)
        classifyTarget.add(1);
      else
        classifyTarget.add(0);
    }
    pool.addTarget(TargetMeta.create("Match", "",  FeatureMeta.ValueType.INTS), classifyTarget.build());
    final int[][] cvSplit = DataTools.splitAtRandom(pool.size(), rng, 0.5, 0.5);
    final Pair<? extends Pool, ? extends Pool> cv = Pair.create(pool.sub(cvSplit[0]), pool.sub(cvSplit[1]));
    final BFGrid grid = GridTools.medianGrid(cv.first.vecData(), 32);
    final GradientBoosting<LLLogit> boosting = new GradientBoosting<>(
        new BootstrapOptimization<>(new GreedyObliviousTree<>(grid, 6), rng),
        LOOL2.class, 2000, 0.02
    );
    final ScoreCalcer learnListener = new ScoreCalcer(/*"\tlearn:\t"*/"\t", cv.first.vecData(), cv.first.target(LLLogit.class));
    final ScoreCalcer validateListener = new ScoreCalcer(/*"\ttest:\t"*/"\t", cv.second.vecData(), cv.second.target(LLLogit.class));
    final ScoreCalcer validatePrec = new ScoreCalcer(/*"\ttest:\t"*/"\t", cv.second.vecData(), cv.second.target("Match", PLogit.class));
    final ScoreCalcer validateRecall = new ScoreCalcer(/*"\ttest:\t"*/"\t", cv.second.vecData(), cv.second.target("Match", RLogit.class));
    final ScoreCalcer learnPrec = new ScoreCalcer(/*"\ttest:\t"*/"\t", cv.first.vecData(), cv.first.target("Match", PLogit.class));
    final ScoreCalcer learnRecall = new ScoreCalcer(/*"\ttest:\t"*/"\t", cv.first.vecData(), cv.first.target("Match", RLogit.class));
    final Consumer<Trans> newLine = trans -> System.out.println();
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(validatePrec);
    boosting.addListener(validateRecall);
    boosting.addListener(learnPrec);
    boosting.addListener(learnRecall);
    boosting.addListener(newLine);
    //noinspection unchecked
    final Ensemble<ObliviousTree> ensemble = boosting.fit(cv.first.vecData(), (LLLogit) cv.first.target(LLLogit.class));
    final ModelTools.CompiledOTEnsemble compile = ModelTools.compile(ensemble);
    double[] scores = new double[grid.rows()];
    for (int f = 0; f < grid.rows(); f++) {
      final BFGrid.Row row = grid.row(f);
      final List<ModelTools.CompiledOTEnsemble.Entry> entries = compile.getEntries();
      for (int i = 0; i < entries.size(); i++) {
        ModelTools.CompiledOTEnsemble.Entry entry = entries.get(i);
        boolean isRelevant = false;
        for (int bfIndex : entry.getBfIndices()) {
          if (bfIndex >= row.start() && bfIndex < row.end()) {
            isRelevant = true;
            break;
          }
        }
        if (isRelevant)
          scores[f] += Math.abs(entry.getValue());
      }
    }
    final int[] order = ArrayTools.sequence(0, scores.length);
    ArrayTools.parallelSort(scores, order);
    for(int i = 0; i < order.length; i++) {
      System.out.println(order[i] + "\t" + scores[i]);

    }
  }

  protected static class ScoreCalcer implements ProgressHandler {
    final String message;
    final Vec current;
    private final VecDataSet ds;
    private final TargetFunc target;

    public ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target) {
      this.message = message;
      this.ds = ds;
      this.target = target;
      current = new ArrayVec(ds.length());
    }

    double min = 1e10;

    @Override
    public void accept(final Trans partial) {
      if (partial instanceof Ensemble) {
        final Ensemble linear = (Ensemble) partial;
        final Trans increment = linear.last();
        for (int i = 0; i < ds.length(); i++) {
          if (increment instanceof Ensemble) {
            current.adjust(i, linear.wlast() * (increment.trans(ds.data().row(i)).get(0)));
          } else {
            current.adjust(i, linear.wlast() * ((Func) increment).value(ds.data().row(i)));
          }
        }
      } else {
        for (int i = 0; i < ds.length(); i++) {
          current.set(i, ((Func) partial).value(ds.data().row(i)));
        }
      }
      final double value = target.value(current);
      System.out.print(message + value);
      min = Math.min(value, min);
      System.out.print(" best = " + min);
    }
  }
}
