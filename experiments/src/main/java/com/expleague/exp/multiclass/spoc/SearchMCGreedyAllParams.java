package com.expleague.exp.multiclass.spoc;

import com.expleague.commons.io.StreamTools;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.*;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.BFGrid;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.loss.multiclass.util.ConfusionMatrix;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.TargetMeta;
import com.expleague.ml.meta.items.QURLItem;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.multiclass.spoc.SPOCMethodClassic;
import com.expleague.ml.methods.multiclass.spoc.impl.CodingMatrixLearningGreedy;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.multiclass.MCModel;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import org.apache.commons.cli.MissingArgumentException;

import java.io.*;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.*;
import java.util.stream.IntStream;

/**
 * User: qdeee
 * Date: 18.08.14
 */
public class SearchMCGreedyAllParams {
  private final static int UNITS = Runtime.getRuntime().availableProcessors();

  public static void main(String[] args) throws MissingArgumentException, IOException {
    final Properties properties = new Properties();
    properties.load(new FileInputStream(new File(args[0])));

    final String mxPath = properties.getProperty("sim_mx_path");
    final CharSequence mxStr = StreamTools.readStream(new BufferedInputStream(new FileInputStream(mxPath)));
    final Mx S = MathTools.CONVERSION.convert(mxStr, Mx.class);

    final String[] strBorders = properties.getProperty("borders").split(";");
    final TDoubleList borders = new TDoubleArrayList();
    for (String strBorder : strBorders) {
      borders.add(Double.valueOf(strBorder));
    }

    final String learnPath = properties.getProperty("learn_path");
    final Pool<QURLItem> learn = DataTools.loadFromFeaturesTxt(learnPath);
    final IntSeq learnTarget = MCTools.transformRegressionToMC(learn.target(L2.class).target, borders.size(), borders);
    learn.addTarget(TargetMeta.create("path", "", FeatureMeta.ValueType.INTS), learnTarget);

    final String testPath = properties.getProperty("test_path");
    final Pool<QURLItem> test = DataTools.loadFromFeaturesTxt(testPath);
    final IntSeq testTarget = MCTools.transformRegressionToMC(test.target(L2.class).target, borders.size(), borders);
    test.addTarget(TargetMeta.create("path", "", FeatureMeta.ValueType.INTS), testTarget);

    final String[] strBaselineScores = properties.getProperty("baseline_scores").split(";");
    final TDoubleList baselineScores = new TDoubleArrayList();
    for (String strBaselineScore : strBaselineScores) {
      baselineScores.add(Double.valueOf(strBaselineScore));
    }
    BaselineComparator.init(baselineScores.toArray());

    final int itersFrom = Integer.valueOf(properties.getProperty("iters_from"));
    final int itersTo = Integer.valueOf(properties.getProperty("iters_to"));
    final int itersDelta = Integer.valueOf(properties.getProperty("iters_delta"));

    final double stepFrom = Double.valueOf(properties.getProperty("step_from"));
    final double stepTo = Double.valueOf(properties.getProperty("step_to"));
    final double stepDelta = Double.valueOf(properties.getProperty("step_delta"));

    final PrintWriter printWriter = new PrintWriter(new FileWriter(properties.getProperty("log_filename")), true);


    final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
    learn.vecData().cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

//    final ThreadPoolExecutor executor = new ThreadPoolExecutor(UNITS, UNITS, 120, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
    final List<Callable<String>> tasks = new LinkedList<>();
    final int k = borders.size();
    for (double lambdaC = 1.0; lambdaC < 1.5 * k; lambdaC += 1.0) {
      for (double lambdaR = 0.5; lambdaR < 3.0; lambdaR += 0.5) {
        for (double lambda1 = 1.0; lambda1 < 1.5 * k; lambda1 += 1.0) {
          for (int iters = itersFrom; iters < itersTo; iters += itersDelta) {
            for (double step = stepFrom; step < stepTo; step += stepDelta) {
              tasks.add(new Task(learn, test, grid, S, k, iters, step, lambdaC, lambdaR, lambda1));
            }
          }
        }
      }
    }
    Collections.shuffle(tasks);

    final ExecutorService pool = Executors.newFixedThreadPool(UNITS);
    final ExecutorCompletionService<String> completionService = new ExecutorCompletionService<>(pool);

    double time = System.currentTimeMillis();

    for (Callable<String> task : tasks) {
      completionService.submit(task);
    }

    pool.shutdown();

    while (!pool.isTerminated()) {
      final String result;
      try {
        result = completionService.take().get();
        printWriter.append(result);
      } catch (InterruptedException | ExecutionException e) {
        e.printStackTrace();
      }
    }
    System.out.println((System.currentTimeMillis() - time) / 1000);
    printWriter.close();
  }

  private static final class Task implements Callable<String> {
    private final Pool<?> learn;
    private final Pool<?> test;
    private final BFGrid grid;
    private final Mx S;

    private final int classCount;

    private final int iters;
    private final double step;

    private final double lac;
    private final double lar;
    private final double la1;


    private Task(final Pool<?> learn, final Pool<?> test, final BFGrid grid, final Mx s, final int classCount,
                 final int iters, final double step, final double lac, final double lar, final double la1) {
      this.learn = learn;
      this.test = test;
      this.grid = grid;
      this.S = s;
      this.classCount = classCount;
      this.iters = iters;
      this.step = step;
      this.lac = lac;
      this.lar = lar;
      this.la1 = la1;
    }

    @Override
    public String call() throws Exception {
      final VecOptimization<LLLogit> weak = (learn, llLogit) -> {
        final GradientBoosting<LLLogit> boosting = new GradientBoosting<>(new GreedyObliviousTree<>(grid, 5), SatL2.class, iters, step);
        final Ensemble ensemble = boosting.fit(learn, llLogit);
        //noinspection unchecked
        return new FuncEnsemble(IntStream.range(0, ensemble.size()).<Trans>mapToObj(ensemble::model).map(f -> (Func) f).toArray(Func[]::new), ensemble.weights());
      };

      final CodingMatrixLearningGreedy codingMatrixLearning = new CodingMatrixLearningGreedy(classCount, 5, lac, lar, la1);
      final Mx mxB = codingMatrixLearning.findMatrixB(S);
      final SPOCMethodClassic spocMethodClassic = new SPOCMethodClassic(mxB, weak);
      final MCModel model = spocMethodClassic.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));

      final Vec predict = model.bestClassAll(test.vecData().data());
      final ConfusionMatrix cm = new ConfusionMatrix(
          test.target(BlockwiseMLLLogit.class).labels(),
          VecTools.toIntSeq(predict));

      final double[] scores = {cm.getMicroPrecision(), cm.getMacroPrecision(), cm.getMacroRecall(), cm.getMacroF1Measure()};
      if (BaselineComparator.getInstance().isBetterThanBaseline(scores)) {
        return String.format("i=%d, s=%f, lac=%f, lar=%f, la1=%f : %.6f | %.6f | %.6f | %.6f\n",
            iters, step, lac, lar, la1, scores[0], scores[1], scores[2], scores[3]);
      }
      else
        return String.format("i=%d, s=%f, lac=%f, lar=%f, la1=%f : fail\n",
            iters, step, lac, lar, la1);
    }
  }

  private static class BaselineComparator {
    private static BaselineComparator instance = null;

    private double[] baselineScores;

    private BaselineComparator(double[] scores){
      this.baselineScores = scores;
    }

    public static void init(final double[] baselineScores) {
      instance = new BaselineComparator(baselineScores);
    }
    public static BaselineComparator getInstance() {
      return instance;
    }

    public boolean isBetterThanBaseline(final double[] scores) {
      return scores[0] > baselineScores[0] ||
          scores[1] > baselineScores[1] ||
          scores[2] > baselineScores[2] |
          scores[3] > baselineScores[3];
    }
  }
}
