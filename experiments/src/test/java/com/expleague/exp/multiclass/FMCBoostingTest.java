package com.expleague.exp.multiclass;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.GridTools;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.factorization.impl.StochasticALS;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.loss.multiclass.util.ConfusionMatrix;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.TargetMeta;
import com.expleague.ml.methods.multiclass.gradfac.FMCBoosting;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.MultiClassModel;
import com.expleague.ml.testUtils.TestResourceLoader;
import junit.framework.TestCase;

import java.io.*;
import java.nio.file.Paths;

/**
 * User: qdeee
 * Date: 27.02.15
 */
public class FMCBoostingTest extends TestCase {
    private static Pool<?> learn;
    private static Pool<?> test;

    private static final String datasetName = "letter";
    private static final String datasetPath = "multiclass/ds_letter/letter.tsv.gz";
    private static final String logDirAbsolutePath = null;

    private synchronized static void init() throws IOException {
        if (learn == null || test == null) {
            final Pool<?> pool = TestResourceLoader.loadPool(datasetPath);
            pool.addTarget(TargetMeta.create(datasetName, "", FeatureMeta.ValueType.INTS),
                    VecTools.toIntSeq(pool.target(L2.class).target)
            );
            final int[][] idxs = DataTools.splitAtRandom(pool.size(), new FastRandom(100500), 0.7, 0.3);
            learn = pool.sub(idxs[0]);
            test = pool.sub(idxs[1]);
        }
    }

    @Override
    protected void setUp() throws Exception {
        init();
    }

    public void testFMCBoostSALS() throws IOException {
        FastRandom rng = new FastRandom(0);
        final FMCBoosting boosting = new FMCBoosting(
                new StochasticALS(rng, 100, 2000, new StochasticALS.Cache(600, 0.01, rng)),
                new GreedyObliviousTree<AdditiveLoss>(GridTools.medianGrid(learn.vecData(), 32), 5),
                L2.class,
                2500,
                5
        );

        fitModel(boosting);
    }


    private void fitModel(final FMCBoosting boosting) throws IOException {
        final FMCBProgressPrinter progressPrinter = new FMCBProgressPrinter(learn, test, logDirAbsolutePath);
        boosting.addListener(progressPrinter);

        long startTime = System.currentTimeMillis();
        final Ensemble ensemble = boosting.fit(learn);
        System.setOut(progressPrinter.getEvaluationOutput());

//    Interval.setStart(startTime);
//    Interval.stopAndPrint("Fitting model");

        final Trans joined = ensemble.last() instanceof FuncJoin ? MCTools.joinBoostingResult(ensemble) : ensemble;
        final MultiClassModel multiclassModel = new MultiClassModel(joined);

//    Interval.start();
        final String learnResult = MCTools.evalModel(multiclassModel, learn, " [LEARN] ", false);
        //Interval.stopAndPrint("Evaluation on LEARN");

        //Interval.start();
        final String testResult = MCTools.evalModel(multiclassModel, test, " [TEST] ", false);
        //Interval.stopAndPrint("Evaluation on TEST");

        System.out.println(learnResult);
        System.out.println(testResult);

        System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out)));
        progressPrinter.finish();
    }

    private static class FMCBProgressPrinter implements ProgressHandler {
        private final VecDataSet learn;
        private final VecDataSet test;

        private final BlockwiseMLLLogit learnMLLLogit;
        private final BlockwiseMLLLogit testMLLLogit;
        private final Mx learnValues;
        private final Mx testValues;

        private final int itersForOut;

        private final PrintStream iterativeOutput;
        private final PrintStream evaluationOutput;
        private final boolean stdout;

        int iteration = 0;
        long lastOutTime = 0;

        public FMCBProgressPrinter(final Pool<?> learn, final Pool<?> test, final String logPath) throws IOException {
            this(learn, test, 10, logPath);
        }

        public FMCBProgressPrinter(final Pool<?> learn, final Pool<?> test, final int itersForOut, final String absoluteLogPath) throws IOException {
            this.learn = learn.vecData();
            this.test = test.vecData();

            this.learnMLLLogit = learn.target(BlockwiseMLLLogit.class);
            this.testMLLLogit = test.target(BlockwiseMLLLogit.class);
            assert learnMLLLogit.classesCount() == testMLLLogit.classesCount();

            this.learnValues = new VecBasedMx(learn.size(), learnMLLLogit.classesCount() - 1);
            this.testValues = new VecBasedMx(test.size(), testMLLLogit.classesCount() - 1);
            this.itersForOut = itersForOut;

            if (absoluteLogPath != null) {
                File outputDir = new File(absoluteLogPath);
                if (outputDir.exists()) {
                    throw new IOException("Directory " + absoluteLogPath + " already exists, try to choose another name for your log file");
                }

                outputDir.mkdirs();
                this.iterativeOutput = new PrintStream(new FileOutputStream(Paths.get(absoluteLogPath, "log.csv").toFile()));
                this.evaluationOutput = new PrintStream(new FileOutputStream(Paths.get(absoluteLogPath, "results.txt").toFile()));
                stdout = false;
            } else {
                this.iterativeOutput = System.out;
                this.evaluationOutput = System.out;
                stdout = true;
            }

            //this.iterativeOutput.println("iteration,train_mP,train_MP,train_MR,test_mP,test_MP,test_MR,iteration_duration");
        }

        @Override
        public void accept(final Trans partial) {
            if (partial instanceof Ensemble) {
                final Ensemble ensemble = (Ensemble) partial;
                final double step = ensemble.wlast();
                final Trans model = ensemble.last();

                VecTools.append(learnValues, VecTools.scale(model.transAll(learn.data()), step));
                VecTools.append(testValues, VecTools.scale(model.transAll(test.data()), step));
            }

            iteration++;
            if (iteration % itersForOut == 0) {
                long iteration_duration = (System.currentTimeMillis() - lastOutTime) / itersForOut;

                final IntSeq learnPredicted;
                final IntSeq testPredicted;

                learnPredicted = convertTransResults(learnValues);
                testPredicted = convertTransResults(testValues);


                //iterativeOutput.print(iteration);

                final ConfusionMatrix learnConfusionMatrix = new ConfusionMatrix(learnMLLLogit.labels(), learnPredicted);
                double lmP = learnConfusionMatrix.getMicroPrecision();
                double lMP = learnConfusionMatrix.getMacroPrecision();
                double lMR = learnConfusionMatrix.getMacroRecall();
                //iterativeOutput.print("," + lmP + "," + lMP + "," + lMR);

                final ConfusionMatrix testConfusionMatrix = new ConfusionMatrix(testMLLLogit.labels(), testPredicted);
                double tmP = testConfusionMatrix.getMicroPrecision();
                double tMP = testConfusionMatrix.getMacroPrecision();
                double tMR = testConfusionMatrix.getMacroRecall();
                //iterativeOutput.print("," + tmP + "," + tMP + "," + tMR);

                //iterativeOutput.println("," + iteration_duration);

                if (!stdout) {
                    String result = String.format("iteration = %d\t\tlearn mP = %.6f\t\t test mP = %.6f", iteration, lmP, tmP);
                    System.out.println(result);
                }

                lastOutTime = System.currentTimeMillis();
            }
        }

        public PrintStream getEvaluationOutput() {
            return evaluationOutput;
        }

        public void finish() {
            iterativeOutput.flush();
            evaluationOutput.flush();
            if (!stdout) {
                iterativeOutput.close();
                evaluationOutput.close();
            }
        }

        private static IntSeq convertTransResults(final Mx trans) {
            final int[] result = new int[trans.rows()];
            for (int i = 0; i < trans.rows(); i++) {
                final Vec row = trans.row(i);
                final int bestClass = VecTools.argmax(row);
                result[i] = row.get(bestClass) > 0 ? bestClass : row.dim();
            }
            return new IntSeq(result);
        }
    }

}