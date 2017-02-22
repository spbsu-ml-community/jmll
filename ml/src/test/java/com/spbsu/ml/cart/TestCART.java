package com.spbsu.ml.cart;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Processor;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.cart.CARTTreeOptimization;
import com.spbsu.ml.methods.cart.CARTTreeOptimizationFixError;
import org.junit.Test;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;

import static java.lang.Math.exp;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;


/**
 * Created by n_buga on 16.10.16.
 */

///home/n_buga/git_repository/jmll/ml/src/test/resources/com/spbsu/ml/allstate_learn.csv

public class TestCART {
    private static final String LearnAllstateFileName = "allstate_learn.csv.gz";
    private static final String TestAllstateFileName = "allstate_test.csv.gz";
    private static final String LearnHIGGSFileName = "HIGGS_learn_1M.csv.gz";
    private static final String TestHIGGSFileName = "HIGGS_test.csv.gz";
    private static final String dir = "src/test/resources/com/spbsu/ml";
    private static final FastRandom rnd = new FastRandom(0);
    private static final String TestBaseDataName = "featuresTest.txt.gz";
    private static final String LearnBaseDataName = "features.txt.gz";

    private Random r = new Random();

    double nextDouble(double range1, double range2) {
        return range1 + r.nextDouble()*(range2 - range1);
    }
/*
    @Test
    public void TestSimple1Dim() { // i -> i^2
        final CARTTreeOptimization testObj = new CARTTreeOptimization();
        int k = 10;
        double[] data = new double[k];
        Vec target = new ArrayVec(k);
        for (int i = 0; i < k; i++) {
            data[i] = i;
            target.set(i, i*i);
        }

        Mx mx = new ColMajorArrayMx(k, data);
        VecDataSet learn = new VecDataSetImpl(mx, null);
        L2 func = new L2(target, learn);
        Trans f = testObj.fit(learn, func);

        double disp = 0;
        double[] quest = new double[k];
        for (int i = 0; i < k/2; i++) {
            quest[i] = i + 0.5;
            double ans = f.compute(new ArrayVec(quest, i, 1)).get(0);
            disp += Math.pow((ans - quest[i]*quest[i]), 2);
        }

        for (int i = k/2; i < k; i++) {
            quest[i] = i;
            double ans = f.compute(new ArrayVec(quest, i, 1)).get(0);
            disp += Math.pow((ans - quest[i]*quest[i]), 2);
        }

        System.out.println(disp/k);

*/
/*        for (int i = 0; i < k; i++) {
            System.out.printf("%f ", quest[i]);
            System.out.println(f.compute(new ArrayVec(quest, i, 1)).get(0));
        } *//*

    }

    @Test
    public void TestSimple2Dim() {
        final CARTTreeOptimization testObj = new CARTTreeOptimization(); // i^2, sqrt i -> i
        int k = 10;
        double[] data = new double[2*k];
        Vec target = new ArrayVec(k);
        for (int i = 0; i < k; i++) {
            target.set(i, i);
            data[i] = i*i;
            data[k + i] = Math.sqrt(i);
        }

        Mx mx = new ColMajorArrayMx(k, data);
        VecDataSet learn = new VecDataSetImpl(mx, null);
        L2 func = new L2(target, learn);
        Trans f = testObj.fit(learn, func);

        double disp = 0;
        double quest[] = new double[2*k];
        for (int i = 0; i < k; i++) {
            Random r = new Random();
            double a = r.nextDouble()*k;
            quest[2*i] = a*a;
            quest[2*i + 1] = Math.sqrt(a);
            double ans = f.compute(new ArrayVec(quest, 2*i, 2)).get(0);
            disp += Math.pow((ans - a), 2);
        }

        System.out.println(disp/k);

*/
/*        for (int i = 0; i < k; i++) {
            System.out.printf("%f %f", quest[2*i], quest[2*i + 1]);
            System.out.println(f.compute(new ArrayVec(quest, 2*i, 2)));
        } *//*

    }

*/
/*    @Test
    public void testRandom2Dim() {
        CARTTreeOptimization testObj = new CARTTreeOptimization();
        int k = 10;
        double[] data = new double[2*k];
        Vec target = new ArrayVec(k);
        int max_bound = 50;
        int min_bound = -50;
        for (int i = 0; i < k; i++) {
            target.set(i, nextDouble(min_bound, max_bound));
            data[i] = nextDouble(min_bound, max_bound);
            data[k + i] = nextDouble(min_bound, max_bound);
        }

        Mx mx = new ColMajorArrayMx(k, data);
        VecDataSet learn = new VecDataSetImpl(mx, null);
        L2 func = new L2(target, learn);
        Trans f = testObj.fit(learn, func);

        double quest[] = new double[2*k];
        for (int i = 0; i < k; i++) {
            Random r = new Random();
            double a = r.nextDouble()*k;
            quest[2*i] = a;
            quest[2*i + 1] = a*a;
        }

        for (int i = 0; i < k; i++) {
            System.out.printf("%f %f", quest[2*i], quest[2*i + 1]);
            System.out.println(f.compute(new ArrayVec(quest, 2*i, 2)));
        }
    } */


    @Test
    public void testnDim() { //function majority
        int n = 3;
        int k = 10;
        Vec data[] = new Vec[k];
        Vec target = new ArrayVec(k);

        data[0] = new ArrayVec(new double[] {0, 1, 0});
        data[1] = new ArrayVec(new double[] {1, 1, 1});
        data[2] = new ArrayVec(new double[] {0, 0, 1});
        data[3] = new ArrayVec(new double[] {0, 0, 0});
        data[4] = new ArrayVec(new double[] {1, 1, 1});
        data[5] = new ArrayVec(new double[] {0 ,0, 0});
        data[6] = new ArrayVec(new double[] {1, 0, 1});
        data[7] = new ArrayVec(new double[] {0, 1, 0});
        data[8] = new ArrayVec(new double[] {0, 0, 0});
        data[9] = new ArrayVec(new double[] {0, 1, 1});

        for (int i = 0; i < k; i++) {
            int sum_1 = 0;
            for (int j = 0; j < n; j++) {
                sum_1 += data[i].get(j);
            }
            if (sum_1 > n/2) {
                target.set(i, 1);
            } else {
                target.set(i, 0);
            }
        }

        Mx mx = new RowsVecArrayMx(data);
        VecDataSet learn = new VecDataSetImpl(mx, null);

        final CARTTreeOptimizationFixError testObj = new CARTTreeOptimizationFixError(learn);

        int weights[] = new int[learn.length()];
        Arrays.fill(weights, 1);
        WeightedLoss func = new WeightedLoss(new L2(target, learn), weights);
        Trans f = testObj.fit(learn, func);

        double disp = 0;
        double quest[] = new double[n*k];
        for (int i = 0; i < k; i++) {
            int sum1 = 0;
            for (int j = 0; j < n; j++) {
                quest[i*n + j] = r.nextInt(2);
                sum1 += quest[i*n + j];
            }
            double ans = f.compute(new ArrayVec(quest, i * n, n)).get(0);
            double real_ans = 0;
            if (sum1 > n/2) {
                real_ans = 1;
            }
            disp += Math.pow((ans - real_ans), 2);
        }

        System.out.println(disp/k);
    }

    @Test
    public void testnDimRand() { //function majority
        int n = 10;
        int k = 201;
        Vec data[] = new Vec[k];
        Vec target = new ArrayVec(k);

        for (int i = 0; i < k; i++) {
            int sum1 = 0;
            data[i] = new ArrayVec(n);
            for (int j = 0; j < n; j++) {
                data[i].set(j, r.nextInt(2));
                sum1 += data[i].get(j);
            }
            if (sum1 > n/2) {
                target.set(i, 1);
            } else {
                target.set(i, 0);
            }
        }

        Mx mx = new RowsVecArrayMx(data);
        VecDataSet learn = new VecDataSetImpl(mx, null);

        final CARTTreeOptimization testObj = new CARTTreeOptimization(learn);

        int weights[] = new int[learn.length()];
        Arrays.fill(weights, 1);
        WeightedLoss func = new WeightedLoss(new L2(target, learn), weights);
        Trans f = testObj.fit(learn, func);

        double disp = 0;
        double quest[] = new double[n*k];
        for (int i = 0; i < k; i++) {
            int cnt = 0;
            for (int j = 0; j < n; j++) {
                quest[i*n + j] = r.nextInt(2);
                cnt += quest[i*n + j];
            }
            double ans = f.compute(new ArrayVec(quest, i*n, n)).get(0);
            int real_ans;
            if (cnt > n/2) {
                real_ans = 1;
            } else {
                real_ans = 0;
            }
            disp += Math.pow((real_ans - ans), 2);
        }

        disp /= k;

        System.out.println(disp);

        for (int i = 0; i < k; i++) {
            double ans = f.compute(data[i]).get(0);
            double right_ans = target.get(i);
//            assert(Math.abs(ans - right_ans) <= testObj.getMaxError());
        }
    }

    private static class AUCCalcer implements ProgressHandler {
        private final String message;
        private final Vec current;
        private  final VecDataSet ds;
        private final Vec rightAns;
        private int allNegative = 0;
        private int allPositive = 0;
        private boolean isWrite = true;

        public AUCCalcer(final String message, final VecDataSet ds, final Vec rightAns) {
            this(message, ds, rightAns, true);
        }

        public AUCCalcer(final String message, final VecDataSet ds, final Vec rightAns, boolean isWrite) {
            this.message = message;
            this.ds = ds;
            this.isWrite = isWrite;
            this.rightAns = rightAns;
            current = new ArrayVec(ds.length());
            for (int i = 0; i < rightAns.dim(); i++) {
                if (rightAns.at(i) == 1) {
                    allPositive += 1;
                } else {
                    allNegative += 1;
                }
            }
        }

        private double max = 0;

        public double getMax() {
            return max;
        }

        private int[] getOrdered(Vec array) {
            int[] order = ArrayTools.sequence(0, array.dim());
            ArrayTools.parallelSort(array.toArray().clone(), order);
            return order;
        }

        @Override
        public void invoke(final Trans partial) {
            int length = ds.length();

            if (partial instanceof Ensemble) {
                final Ensemble linear = (Ensemble) partial;
                final Trans increment = linear.last();
                for (int i = 0; i < length; i++) {
                    if (increment instanceof Ensemble) {
                        current.adjust(i, linear.wlast() * (increment.trans(ds.data().row(i)).get(0)));
                    } else {
                        current.adjust(i, linear.wlast() * ((Func) increment).value(ds.data().row(i)));
                    }
                }
            }
            else {
                for (int i = 0; i < length; i++) {
                    current.set(i, ((Func) partial).value(ds.data().row(i)));
                }
            }

            int ordered[] = getOrdered(current);
            int trueNegative = 0;
            int falseNegative = 0;

            double sum = 0;
            int curPos = 0;

            double prevFPR = 1;

//            ArrayList<Double> x = new ArrayList<>();
//            ArrayList<Double> y = new ArrayList<>();

            while (curPos < ordered.length) {
                if (rightAns.get(ordered[curPos++]) != 1) { //!!!!
                    trueNegative += 1;
                } else {
                    falseNegative += 1;
                    continue;
                }
                double falsePositive = allNegative - trueNegative;
                double truePositive = allPositive - falseNegative;
                double TPR = 1.0*truePositive/allPositive;
                double FPR = 1.0*falsePositive/allNegative;

//                x.add(FPR);
//                y.add(TPR);
                sum += TPR * (prevFPR - FPR);
                prevFPR = FPR;
            }

/*            XYChart ex = org.knowm.xchart.QuickChart.getChart("Simple chart", "x", "y",
                    "y(x)", x, y);
            new org.knowm.xchart.SwingWrapper<>(ex).displayChart(); */

            final double value = sum;
            if (isWrite) System.out.print(message + value);
            max = Math.max(value, max);
            if (isWrite) System.out.print(" best = " + max);
        }
    }

    protected static class ScoreCalcer implements ProgressHandler {
        private final String message;
        private final Vec current;
        private final VecDataSet ds;
        private final TargetFunc target;
        private boolean isWrite = true;

        public ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target) {
            this(message, ds, target, true);
        }

        public ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target, boolean isWrite) {
            this.message = message;
            this.isWrite = isWrite;
            this.ds = ds;
            this.target = target;
            current = new ArrayVec(ds.length());
        }

        double max = 0;
        double min = 1e10;

        public double getMax() {
            return max;
        }

        @Override
        public void invoke(final Trans partial) {
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
            final double valuePerp = exp(-target.value(current)/target.dim());

            if (isWrite) System.out.print(message + valuePerp + " | " + value);
            max = Math.max(valuePerp, max);
            min = Math.min(value, min);
            if (isWrite) System.out.print(" best = " + max + " | " + min);
        }
    }

    private static abstract class TestProcessor implements Processor<CharSequence> {
        protected VecBuilder targetBuilder = new VecBuilder();
        protected VecBuilder featuresBuilder = new VecBuilder();
        protected int featuresCount = -1;

        public VecBuilder getTargetBuilder() {
            return targetBuilder;
        }

        public VecBuilder getFeaturesBuilder() {
            return featuresBuilder;
        }

        public int getFeaturesCount() {
            return featuresCount;
        }

        protected abstract void init();

        public void wipe() {
            targetBuilder = new VecBuilder();
            featuresBuilder = new VecBuilder();
            featuresCount = -1;
            init();
        }
    }

    private static class BaseDataReadProcessor extends TestProcessor {
        protected void init() {};

        @Override
        public void process(CharSequence arg) {
            final CharSequence[] parts = CharSeqTools.split(arg, '\t');
            targetBuilder.append(CharSeqTools.parseDouble(parts[1]));
            if (featuresCount < 0)
                featuresCount = parts.length - 4;
            else if (featuresCount != parts.length - 4)
                throw new RuntimeException("\"Failed to parse line \" + lindex + \":\"");
            for (int i = 4; i < parts.length; i++) {
                featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
            }
        }
    }

    private static class AllstateReadProcessor extends TestProcessor {
        protected void init() {};

        public void addCategoricalParameter(int value, int boundValue) {
            for (int i = 0; i < value && i < boundValue; i++) {
                featuresBuilder.append(0);
            }
            if (value < boundValue)
                featuresBuilder.append(1);
            for (int i = value + 1; i < boundValue; i++) {
                featuresBuilder.append(0);
            }
        }

        @Override
        public void process(CharSequence arg) {
            int curCountFeatures = 0;
            final CharSequence[] parts = CharSeqTools.split(arg, ',');
            targetBuilder.append(CharSeqTools.parseDouble(parts[34]));
            featuresBuilder.append(CharSeqTools.parseInt(parts[3]));
            curCountFeatures += 1;
            featuresBuilder.append(CharSeqTools.parseInt(parts[4]));
            curCountFeatures += 1;

            String vehicleMake = parts[5].toString();
            int value = 0;
            for (int i = 0; i < vehicleMake.length(); i++) {
                value *= 24;
                value += (int)vehicleMake.charAt(i) - (int)'A';
            }
            addCategoricalParameter(value, 78);
            curCountFeatures += 78;

            int[] boundValues = {11, 4, 7, 4, 4, 7, 5, 4, 2, 4, 7, 7};
            for (int i = 0; i < 12; i++) {
                int curCat = (int)parts[8 + i].charAt(0) - (int)'A';
                addCategoricalParameter(curCat, boundValues[i]);
                curCountFeatures += boundValues[i];
            }

            for (int i = 20; i <= 28; i++) {
                featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
                curCountFeatures += 1;
            }

            int nvCat = (int)parts[29].charAt(0) - (int)'A';
            addCategoricalParameter(nvCat, 15);
            curCountFeatures += 15;

            for (int i = 30; i <= 33; i++) {
                featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
                curCountFeatures += 1;
            }
            if (featuresCount == -1) {
                featuresCount = curCountFeatures;
            } else {
                assert(featuresCount == curCountFeatures);
            }
        }
    }

    private class HIGGSReadProcessor extends TestProcessor {
        protected void init() { featuresCount = 28; }

        public HIGGSReadProcessor() {
            init();
        }

        @Override
        public void process(CharSequence arg) {
            final CharSequence[] parts = CharSeqTools.split(arg, ',');
            int curAns = (int)CharSeqTools.parseDouble(parts[0]);
            if (curAns == 0) curAns = -1;
            targetBuilder.append(curAns);
            for (int i = 1; i <= 28; i++) {
                featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
            }
        }
    }

/*    private static class BestPerplexityCalcer implements ProgressHandler {
        final String message = "Best perplexity";
        final Vec current;
        private final VecDataSet ds;
        private final TargetFunc target;

        public BestPerplexityCalcer(final VecDataSet ds, final TargetFunc target) {
            this.ds = ds;
            this.target = target;
            current = new ArrayVec(ds.length());
        }

        double max = 0;
        int counter = 0;
        int stepNumber = 0;

        @Override
        public void invoke(final Trans partial) {
            counter++;
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
            final double valuePerp = exp(-target.value(current) / target.dim());

//            System.out.print(message + valuePerp + " | " + value);
            if (max < valuePerp) {
                max = Math.max(valuePerp, max);
                stepNumber = counter;
            }
        }

        public double getMax() {
            return max;
        }
    } */

    private static String getFullPath(String file) {
        return Paths.get(System.getProperty("user.dir"), dir, file).toString();
    }

    private static class DataML {
        final private VecDataSet learnFeatures;
        final private Vec learnTarget;
        final private VecDataSet testFeatures;
        final private Vec testTarget;

        public DataML(VecDataSet learnFeatures, Vec learnTarget, VecDataSet testFeatures, Vec testTarget) {
            this.learnFeatures = learnFeatures;
            this.learnTarget = learnTarget;
            this.testFeatures = testFeatures;
            this.testTarget = testTarget;
        }

        public VecDataSet getLearnFeatures() {
            return learnFeatures;
        }

        public Vec getLearnTarget() {
            return learnTarget;
        }

        public VecDataSet getTestFeatures() {
            return testFeatures;
        }

        public Vec getTestTarget() {
            return testTarget;
        }
    }

    private static Reader getReader(String fileName) throws IOException {
        return fileName.endsWith(".gz") ?
                new InputStreamReader(new GZIPInputStream(new FileInputStream(getFullPath(fileName)))) :
                new FileReader(getFullPath(fileName));
    }

    private static DataML readData(TestProcessor processor, String learnFileName, String testFileName)
            throws IOException {
        final Reader inLearn = getReader(learnFileName);
        final Reader inTest = getReader(testFileName);

        CharSeqTools.processLines(inLearn, processor);
        Mx data = new VecBasedMx(processor.getFeaturesCount(),
                processor.getFeaturesBuilder().build());
        VecDataSet learnFeatures = new VecDataSetImpl(data, null);
        Vec learnTarget = processor.getTargetBuilder().build();
        processor.wipe();
        CharSeqTools.processLines(inTest, processor);
        data = new VecBasedMx(processor.getFeaturesCount(),
                processor.getFeaturesBuilder().build());
        VecDataSet testFeatures = new VecDataSetImpl(data, null);
        Vec testTarget = processor.getTargetBuilder().build();
        return new DataML(learnFeatures, learnTarget, testFeatures, testTarget);
    }

    private static class RGBoostRunner {
        private final DataML data;
        private final TargetFunc learnTarget;
        private final TargetFunc testTarget;
        private List<Action<? super Trans>> handlers = new ArrayList<>();
        private int iterationsCount = 10000;
        private double step = 0.002;
        private FuncKind funcKind;

        public enum FuncKind {
            LLLogit,
            L2
        }

        public RGBoostRunner(DataML data, FuncKind funcKind) {
            this.data = data;
            this.funcKind = funcKind;
            if (funcKind == FuncKind.L2) {
/*                final int[] weights = new int[data.getLearnTarget().dim()];
                for (int i = 0; i < weights.length; i++)
                    weights[i] = rnd.nextPoisson(1);
                learnTarget = new WeightedLoss<>(new L2(data.getLearnTarget(), data.getLearnFeatures()), weights); */
                learnTarget = new L2(data.getLearnTarget(), data.getLearnFeatures());
                testTarget = new L2(data.getTestTarget(), data.getTestFeatures());
            } else {
                learnTarget = new LLLogit(data.getLearnTarget(), data.getLearnFeatures());
                testTarget = new LLLogit(data.getTestTarget(), data.getTestFeatures());
            }

            final Action counter = new ProgressHandler() {
                int index = 0;

                @Override
                public void invoke(final Trans partial) {
                    System.out.print("\n" + index++);
                }
            };
            handlers.add(counter);
        }

        public RGBoostRunner setIterations(int iterations) {
            this.iterationsCount = iterations;
            return this;
        }

        public RGBoostRunner setStep(double step) {
            this.step = step;
            return this;
        }

        public RGBoostRunner addScoreCalcerLearn() {
            final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", data.getLearnFeatures(),
                    learnTarget);
            handlers.add(learnListener);
            return this;
        }

        public RGBoostRunner addScoreCalcerTest() {
            final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", data.getTestFeatures(),
                    testTarget);
            handlers.add(validateListener);
            return this;
        }

        public RGBoostRunner addAUCCalcerTest() {
            final AUCCalcer aucCalcerTest = new AUCCalcer("\tAUC test:\t", data.getTestFeatures(),
                    data.getTestTarget());
            handlers.add(aucCalcerTest);
            return this;
        }

        public RGBoostRunner addAUCCalcerLearn() {
            final AUCCalcer aucCalcerLearn = new AUCCalcer("\tAUC learn:\t", data.getLearnFeatures(),
                    data.getLearnTarget());
            handlers.add(aucCalcerLearn);
            return this;
        }

        public void run() {
            if (funcKind == FuncKind.L2) {
                final GradientBoosting<L2> boosting = new GradientBoosting<L2>(
                        new BootstrapOptimization<L2>(
                                new CARTTreeOptimization(data.getLearnFeatures()), rnd), L2.class,
                        iterationsCount, step);
                for (Action<? super Trans> action: handlers) {
                    boosting.addListener(action);
                }

                boosting.fit(data.getLearnFeatures(), (L2)learnTarget);
            } else {
                final GradientBoosting<LLLogit> boosting = new GradientBoosting<LLLogit>(
                        new BootstrapOptimization<L2>(
                                new CARTTreeOptimization(data.getLearnFeatures()), rnd), L2.class,
                        iterationsCount, step);
                for (Action<? super Trans> action: handlers) {
                    boosting.addListener(action);
                }

                boosting.fit(data.getLearnFeatures(), (LLLogit)learnTarget);
            }
        }

        public double findBestTestScore() {
            if (funcKind == FuncKind.L2) {
                final ScoreCalcer bestPerplexityCalcer = new ScoreCalcer("", data.getTestFeatures(),
                        testTarget, false);
                final GradientBoosting<L2> boosting = new GradientBoosting<L2>(
                        new BootstrapOptimization<L2>(
                                new CARTTreeOptimization(data.getLearnFeatures()), rnd), L2.class,
                        iterationsCount, step);

                for (Action<? super Trans> action: handlers) {
                    boosting.addListener(action);
                }

                boosting.addListener(bestPerplexityCalcer);
                boosting.fit(data.getLearnFeatures(), (L2) learnTarget);
                return bestPerplexityCalcer.getMax();
            } else if (funcKind == FuncKind.LLLogit) {
                final AUCCalcer bestAUCCalcer = new AUCCalcer("", data.getTestFeatures(),
                        data.getTestTarget(), false);
                final GradientBoosting<LLLogit> boosting = new GradientBoosting<LLLogit>(
                        new BootstrapOptimization<L2>(
                                new CARTTreeOptimization(data.getLearnFeatures()), rnd), L2.class,
                        iterationsCount, step);
                for (Action<? super Trans> action: handlers) {
                    boosting.addListener(action);
                }

                boosting.addListener(bestAUCCalcer);
                boosting.fit(data.getLearnFeatures(), (LLLogit)learnTarget);
                return bestAUCCalcer.getMax();
            }
            return 0;
        }
    }

    @Test
    public void testGRBoostCARTAllstate() throws IOException {
        TestProcessor processor = new AllstateReadProcessor();
        DataML data = readData(processor, LearnAllstateFileName, TestAllstateFileName);
        RGBoostRunner rgBoostRunner = new RGBoostRunner(data, RGBoostRunner.FuncKind.L2);
        rgBoostRunner
                .addScoreCalcerLearn()
                .addScoreCalcerTest()
                .run();
    }

    @Test
    public void testGRBoostCARTHIGGS() throws IOException {
        TestProcessor processor = new HIGGSReadProcessor();
        DataML data = readData(processor, LearnHIGGSFileName, TestHIGGSFileName);
        RGBoostRunner rgBoostRunner = new RGBoostRunner(data, RGBoostRunner.FuncKind.LLLogit);
        rgBoostRunner
                .addScoreCalcerLearn()
                .addScoreCalcerTest()
                .addAUCCalcerLearn()
                .addAUCCalcerTest()
                .setStep(1.6);

        rgBoostRunner.run();
    }

    @Test
    public void testGRBoostBaseData() throws IOException {
        TestProcessor processor = new BaseDataReadProcessor();
        DataML data = readData(processor, LearnBaseDataName, TestBaseDataName);
        assertEquals(50, data.getLearnFeatures().data().columns());
        assertEquals(50, data.getTestFeatures().data().columns());
        assertEquals(12465, data.getLearnFeatures().data().rows());
        assertEquals(46596, data.getTestFeatures().data().rows());
        RGBoostRunner rgBoostRunner = new RGBoostRunner(data, RGBoostRunner.FuncKind.L2);
        rgBoostRunner.addScoreCalcerLearn()
                .addScoreCalcerTest().
                run();
    }

    DataML bootstrap(DataML data) {
        VecBuilder targetBuilder = new VecBuilder();
        VecBuilder featureBuilder = new VecBuilder();
        int countFeatures = data.getLearnFeatures().xdim();
        for (int i = 0; i < data.getLearnFeatures().length(); i++) {
            int cnt_i = rnd.nextPoisson(1.);
            Vec curVec = data.getLearnFeatures().at(i);
            for (int j = 0; j < cnt_i; j++) {
                for (int k = 0; k < curVec.dim(); k++) {
                    featureBuilder.add(curVec.get(k));
                }
                targetBuilder.append(data.getLearnTarget().get(i));
            }
        }

        Mx dataMX = new VecBasedMx(countFeatures,
                featureBuilder.build());
        VecDataSet learnFeatures = new VecDataSetImpl(dataMX, null);

        return new DataML(learnFeatures, targetBuilder.build(),
                data.getTestFeatures(), data.getTestTarget());
    }

    @Test
    public void testGRBoostBaseDataConfInterval() throws IOException {
        TestProcessor processor = new BaseDataReadProcessor();
        DataML data = readData(processor, LearnBaseDataName, TestBaseDataName);

        final int M = 10;
        double scores[] = new double[M];

        RGBoostRunner rgBoostRunner = new RGBoostRunner(data, RGBoostRunner.FuncKind.L2);
        scores[0] = rgBoostRunner.setIterations(1000).findBestTestScore();
        for (int i = 1; i < M; i++) {
            data = bootstrap(data);
            rgBoostRunner = new RGBoostRunner(data, RGBoostRunner.FuncKind.L2);
            scores[i] = rgBoostRunner.setIterations(1000).findBestTestScore();
        }
        Arrays.sort(scores);
        System.out.printf("\n%.6f %.6f\n", scores[4], scores[M - 5]);
    }

    @Test
    public void testGRBoostHIGGSDataConfInterval() throws IOException {
        TestProcessor processor = new HIGGSReadProcessor();

        DataML data = readData(processor, LearnHIGGSFileName, TestHIGGSFileName);

        final int M = 6;
        double scores[] = new double[M];

        RGBoostRunner rgBoostRunner = new RGBoostRunner(data, RGBoostRunner.FuncKind.LLLogit);
        scores[0] = rgBoostRunner.setIterations(5).findBestTestScore();
        for (int i = 1; i < M; i++) {

            rgBoostRunner = new RGBoostRunner(data, RGBoostRunner.FuncKind.LLLogit);
            scores[i] = rgBoostRunner.setIterations(5).findBestTestScore();
        }
        Arrays.sort(scores);
        System.out.printf("%.4f %.4f\n", scores[4], scores[M - 5]);
    }

/*    @Test
    public void testGRBoostHIGGSDataConfInterval1() throws IOException {
        TestProcessor processor = new HIGGSReadProcessor();
        DataML data = readData(processor, LearnHIGGSFileName, TestHIGGSFileName);

        final int M = 6;
        double scores[] = new double[M];

        RGBoostRunner rgBoostRunner = new RGBoostRunner(data, RGBoostRunner.FuncKind.LLLogit);
        scores[0] = rgBoostRunner.setIterations(5).findBestTestScore();
        for (int i = 1; i < M; i++) {
            data = bootstrap(data);
            rgBoostRunner = new RGBoostRunner(data, RGBoostRunner.FuncKind.LLLogit);
            scores[i] = rgBoostRunner.setIterations(5).findBestTestScore();
        }
        Arrays.sort(scores);
        System.out.printf("%.4f %.4f\n", scores[4], scores[M - 5]);
    } */

}

