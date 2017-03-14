package com.spbsu.exp.multiclass.spoc;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.*;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.multiclass.util.ConfusionMatrix;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.fake.FakeTargetMeta;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.multiclass.spoc.SPOCMethodClassic;
import com.spbsu.ml.methods.multiclass.spoc.impl.CodingMatrixLearning;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.multiclass.MCModel;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import org.apache.commons.cli.MissingArgumentException;

import java.io.*;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.*;

/**
 * User: qdeee
 * Date: 18.08.14
 */
public class SearchMCMathAllParams {
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
    learn.addTarget(new FakeTargetMeta(learn.vecData(), FeatureMeta.ValueType.INTS), learnTarget);

    final String testPath = properties.getProperty("test_path");
    final Pool<QURLItem> test = DataTools.loadFromFeaturesTxt(testPath);
    final IntSeq testTarget = MCTools.transformRegressionToMC(test.target(L2.class).target, borders.size(), borders);
    test.addTarget(new FakeTargetMeta(test.vecData(), FeatureMeta.ValueType.INTS), testTarget);

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

    final String[] strMlsArr = properties.getProperty("mls_arr").split(";");
    final TDoubleList mlsArr = new TDoubleArrayList();
    for (String strMls : strMlsArr) {
      mlsArr.add(Double.valueOf(strMls));
    }
    final double[] mlSteps = mlsArr.toArray();


    final PrintWriter printWriter = new PrintWriter(new FileWriter(properties.getProperty("log_filename")), true);


    final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
    learn.vecData().cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    final List<Callable<String>> tasks = new LinkedList<>();
    final int k = borders.size();
    for (double mls : mlSteps) {
      for (double lambdaC = 1.0; lambdaC < 1.5 * k; lambdaC += 1.0) {
        for (double lambdaR = 0.5; lambdaR < 3.0; lambdaR += 0.5) {
          for (double lambda1 = 1.0; lambda1 < 1.5 * k; lambda1 += 1.0) {
            for (int iters = itersFrom; iters < itersTo; iters += itersDelta) {
              for (double step = stepFrom; step < stepTo; step += stepDelta) {
                tasks.add(new Task(learn, test, grid, S, k, iters, step, lambdaC, lambdaR, lambda1, mls));
              }
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
    private final double mls;


    private Task(final Pool<?> learn, final Pool<?> test, final BFGrid grid, final Mx s, final int classCount,
                 final int iters, final double step, final double lac, final double lar, final double la1, final double mls) {
      this.learn = learn;
      this.test = test;
      this.grid = grid;
      S = s;
      this.classCount = classCount;
      this.iters = iters;
      this.step = step;
      this.lac = lac;
      this.lar = lar;
      this.la1 = la1;
      this.mls = mls;
    }

    @Override
    public String call() throws Exception {
      final VecOptimization<LLLogit> weak = new VecOptimization<LLLogit>() {
        @Override
        public Func fit(final VecDataSet learn, final LLLogit llLogit) {
          final GradientBoosting<LLLogit> boosting = new GradientBoosting<>(new GreedyObliviousTree<L2>(grid, 5), SatL2.class, iters, step);
          final Ensemble ensemble = boosting.fit(learn, llLogit);
          return new FuncEnsemble(ArrayTools.map(ensemble.models, Func.class, new Computable<Trans, Func>() {
            @Override
            public Func compute(final Trans argument) {
              return (Func)argument;
            }
          }), ensemble.weights);
        }
      };

      final CodingMatrixLearning codingMatrixLearning = new CodingMatrixLearning(classCount, 5, lac, lar, la1, mls);
      final Mx mxB = codingMatrixLearning.findMatrixB(S);
      final SPOCMethodClassic spocMethodClassic = new SPOCMethodClassic(mxB, weak);
      final MCModel model = (MCModel) spocMethodClassic.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));

      final Vec predict = model.bestClassAll(test.vecData().data());
      final ConfusionMatrix cm = new ConfusionMatrix(
          test.target(BlockwiseMLLLogit.class).labels(),
          VecTools.toIntSeq(predict));

      final double[] scores = {cm.getMicroPrecision(), cm.getMacroPrecision(), cm.getMacroRecall(), cm.getMacroF1Measure()};
      if (BaselineComparator.getInstance().isBetterThanBaseline(scores)) {
        return String.format("i=%d, s=%f, lac=%f, lar=%f, la1=%f, mls=%f : %.6f | %.6f | %.6f | %.6f\n",
            iters, step, lac, lar, la1, mls, scores[0], scores[1], scores[2], scores[3]);
      }
      else
        return String.format("i=%d, s=%f, lac=%f, lar=%f, la1=%f, mls=%f : fail\n",
            iters, step, lac, lar, la1, mls);
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
