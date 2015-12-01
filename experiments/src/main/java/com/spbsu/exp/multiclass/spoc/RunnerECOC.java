package com.spbsu.exp.multiclass.spoc;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.GridTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.cli.builders.data.impl.DataBuilderClassic;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.multiclass.spoc.ECOCCombo;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.multiclass.MCModel;
import com.spbsu.ml.models.multiclass.MulticlassCodingMatrixModel;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * User: qdeee
 * Date: 24.11.14
 */
public class RunnerECOC {
  public static void main(String[] args) throws IOException {
    final Properties properties = new Properties();
    properties.load(new FileInputStream(args[0]));

    final boolean isJsonFormat = properties.getProperty("is_json").equals("true");
    final String learnPath = properties.getProperty("learn_path");
    final String testPath = properties.getProperty("test_path");
    final String mxPath = properties.getProperty("sim_mx_path");

    final int l = Integer.valueOf(properties.getProperty("L", String.valueOf(5)));

    final double lambdaC = Double.valueOf(properties.getProperty("lac", String.valueOf(5.0)));
    final double lambdaR = Double.valueOf(properties.getProperty("lar", String.valueOf(2.5)));
    final double lambda1 = Double.valueOf(properties.getProperty("la1", String.valueOf(3.0)));

    final int iters = Integer.valueOf(properties.getProperty("iters", String.valueOf(100)));
    final double step = Double.valueOf(properties.getProperty("step", String.valueOf(0.3)));

    final boolean updatePrior = properties.getProperty("update_prior").equals("false");
    final boolean targetBasedUpdate = Boolean.valueOf(properties.getProperty("target_based_update", "false"));
    final int firstColumnForUpdate = Integer.valueOf(properties.getProperty("first_column_for_update", "5"));
    final double lambdaPrior = Double.valueOf(properties.getProperty("laprior", String.valueOf(0.8)));

    properties.store(System.out, "[PROPERTIES VALUES] ");

    final CharSequence mxStr = StreamTools.readStream(new FileInputStream(mxPath));
    final Mx S = MathTools.CONVERSION.convert(mxStr, Mx.class);

    final DataBuilderClassic dataBuilder = new DataBuilderClassic();
    dataBuilder.setJsonFormat(isJsonFormat);
    dataBuilder.setLearnPath(learnPath);
    dataBuilder.setTestPath(testPath);
    final Pair<Pool, Pool> poolsPair = dataBuilder.create();
    final Pool<?> learn = poolsPair.getFirst();
    final Pool<?> test = poolsPair.getSecond();
    final VecDataSet vecDataSet = learn.vecData();
    final BFGrid grid = GridTools.medianGrid(vecDataSet, 32);

    final BlockwiseMLLLogit mllLogit = learn.target(BlockwiseMLLLogit.class);
    final int k = MCTools.countClasses(mllLogit.labels());

    final ECOCCombo ecocComboMethod = new ECOCCombo(k, l, lambdaC, lambdaR, lambda1, S, createWeak(grid, iters, step));
    final Action<MulticlassCodingMatrixModel> listener = new Action<MulticlassCodingMatrixModel>() {
      @Override
      public void invoke(final MulticlassCodingMatrixModel model) {
        if (updatePrior && model.getCodingMatrix().columns() >= firstColumnForUpdate) {
          final Mx mx = getPairwiseInteractions(model, learn, targetBasedUpdate);
          VecTools.scale(S, lambdaPrior);
          VecTools.scale(mx, 1 - lambdaPrior);
          VecTools.append(S, mx);
        }

        System.out.println("L == " + model.getInternalModel().ydim());
        System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", true));
        System.out.println(MCTools.evalModel(model, test, "[TEST] ", true));
        System.out.println();
      }
    };
    ecocComboMethod.addListener(listener);
    final MulticlassCodingMatrixModel model = (MulticlassCodingMatrixModel) ecocComboMethod.fit(vecDataSet, mllLogit);
    System.out.println("\n\n\n");
    System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST] ", false));
  }

  private static VecOptimization<LLLogit> createWeak(final BFGrid grid, final int iters, final double step) {
    return new VecOptimization<LLLogit>() {
      @Override
      public Trans fit(final VecDataSet learn, final LLLogit llLogit) {
        final GradientBoosting<LLLogit> boosting = new GradientBoosting<>(new GreedyObliviousTree<L2>(grid, 5), iters, step);
        final Ensemble ensemble = boosting.fit(learn, llLogit);
        return new FuncEnsemble(
            ArrayTools.map(ensemble.models, Func.class, new Computable<Trans, Func>() {
              @Override
              public Func compute(final Trans argument) {
                return (Func) argument;
              }
            }),
            ensemble.weights
        );
      }
    };
  }

  private static Mx getPairwiseInteractions(final MCModel model, final Pool<?> pool, final boolean targetBasedUpdate) {
    final BlockwiseMLLLogit mllLogit = pool.target(BlockwiseMLLLogit.class);
    final VecDataSet ds = pool.vecData();
    final Mx result = new VecBasedMx(mllLogit.classesCount(), mllLogit.classesCount());

    final Mx features = ds.data();
    final int[] counts = new int[features.rows()];
    for (int i = 0; i < ds.length(); i++) {
      final Vec probs = model.probs(features.row(i));
      final int bestClass = targetBasedUpdate ? mllLogit.label(i) : VecTools.argmax(probs);
      VecTools.append(result.row(bestClass), probs);
      counts[bestClass]++;
    }
    for (int c = 0; c < result.rows(); c++) {
      VecTools.scale(result.row(c), 1.0 / counts[c]);
    }
    for (int c1 = 0; c1 < result.rows(); c1++) {
      for (int c2 = c1 + 1; c2 < result.columns(); c2++) {
        final double val = 0.5 * (result.get(c1, c2) + result.get(c2, c1));
        result.set(c1, c2, val);
        result.set(c2, c1, val);
      }
    }
    return result;
  }
}


