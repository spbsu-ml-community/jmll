package com.spbsu.exp.cart;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Processor;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.GridTools;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.LOOL2;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.GradientBoosting;

import java.io.*;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;


/**
 * Created by n_buga on 23.02.17.
 * Estimate the accuracy of the CART tree.
 */
public class ConfidentIntervalFinder {
    private static final String dir = "ml/src/test/resources/com/spbsu/ml";
//    private static final String dir = "./resources";
    private static final String TestBaseDataName = "featuresTest.txt.gz";
    private static final String LearnBaseDataName = "features.txt.gz";
    private static final String LearnCTSliceFileName = "slice_train.csv";
    private static final String TestCTSliceFileName = "slice_test.csv";
    private static final String LearnKSHouseFileName = "learn_ks_house.csv";
    private static final String TestKSHouseFileName = "test_ks_house.csv";
    private static final String LearnHIGGSFileName = "HIGGS_learn_1M.csv.gz";
    private static final String TestHIGGSFileName = "HIGGS_test.csv.gz";
    private static final String LearnCancerFileName = "Cancer_learn.csv";
    private static final String TestCancerFileName = "Cancer_test.csv";
    private static final String ResultFile = "result6.txt";

    private static final FastRandom rnd = new FastRandom(System.currentTimeMillis());

    private static class AUCCalcer implements ProgressHandler {
        private final String message;
        private final Vec current;
        private  final VecDataSet ds;
        private final Vec rightAns;
        private int allNegative = 0;
        private int allPositive = 0;
        private boolean isWrite = true;

        AUCCalcer(final String message, final VecDataSet ds, final Vec rightAns) {
            this(message, ds, rightAns, true);
        }

        AUCCalcer(final String message, final VecDataSet ds, final Vec rightAns, boolean isWrite) {
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

        double getMax() {
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

/*            ArrayList<Double> x = new ArrayList<>();
            ArrayList<Double> y = new ArrayList<>(); */

            double max_accuracy = 0;

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

                double cur_accuracy = 1.0*(trueNegative + truePositive)/(allPositive + allNegative);
                max_accuracy = Math.max(max_accuracy, cur_accuracy);
            }

/*            XYChart ex = org.knowm.xchart.QuickChart.getChart("Simple chart", "x", "y",
                    "y(x)", x, y);
            new org.knowm.xchart.SwingWrapper<>(ex).displayChart(); */

            final double value = sum;
            if (isWrite) System.out.print(message + value);
            max = Math.max(value, max);
            if (isWrite) System.out.print(" best = " + max);

            System.out.printf(" rate = %.5f", max_accuracy);
        }
    }

    private static DataML bootstrap(DataML data) {
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

    private static class DataML {
        final private VecDataSet learnFeatures;
        final private Vec learnTarget;
        final private VecDataSet testFeatures;
        final private Vec testTarget;

        DataML(VecDataSet learnFeatures, Vec learnTarget, VecDataSet testFeatures, Vec testTarget) {
            this.learnFeatures = learnFeatures;
            this.learnTarget = learnTarget;
            this.testFeatures = testFeatures;
            this.testTarget = testTarget;
        }

        VecDataSet getLearnFeatures() {
            return learnFeatures;
        }

        Vec getLearnTarget() {
            return learnTarget;
        }

        VecDataSet getTestFeatures() {
            return testFeatures;
        }

        Vec getTestTarget() {
            return testTarget;
        }
    }

    private static String getFullPath(String file) {
        return Paths.get(System.getProperty("user.dir"), dir, file).toString();
    }

    private static Reader getReader(String fileName) throws IOException {
        return fileName.endsWith(".gz") ?
                new InputStreamReader(new GZIPInputStream(new FileInputStream(getFullPath(fileName)))) :
                new FileReader(getFullPath(fileName));
    }

    private static abstract class TestProcessor implements Processor<CharSequence> {
        public abstract VecBuilder getTargetBuilder();

        public abstract VecBuilder getFeaturesBuilder();

        public abstract int getFeaturesCount();

        public void wipe() {
            getFeaturesBuilder().clear();
            getTargetBuilder().clear();
        }
    }

    private static class KSHouseReadProcessor extends TestProcessor {
        private VecBuilder targetBuilder = new VecBuilder();
        private VecBuilder featuresBuilder = new VecBuilder();
        private int featureCount = 19;

        @Override
        public void process(CharSequence arg) {
            final CharSequence[] parts = CharSeqTools.split(arg, ',');
            targetBuilder.append(CharSeqTools.parseDouble(parts[2]));
            for (int i = 3; i <= 20; i++) {
                featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
            }
            featuresBuilder.append(CharSeqTools.parseDouble(parts[1]));
        }

        @Override
        public VecBuilder getTargetBuilder() {
            return targetBuilder;
        }

        @Override
        public VecBuilder getFeaturesBuilder() {
            return featuresBuilder;
        }

        @Override
        public int getFeaturesCount() {
            return featureCount;
        }
    }

    private static class CTSliceReadProcessor extends TestProcessor {
        private VecBuilder targetBuilder = new VecBuilder();
        private VecBuilder featuresBuilder = new VecBuilder();
        private int featureCount = 384;

        @Override
        public void process(CharSequence arg) {
            final CharSequence[] parts = CharSeqTools.split(arg, ',');
            targetBuilder.append(CharSeqTools.parseDouble(parts[385]));
            for (int i = 1; i < 385; i++) {
                featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
            }
        }

        @Override
        public VecBuilder getTargetBuilder() {
            return targetBuilder;
        }

        @Override
        public VecBuilder getFeaturesBuilder() {
            return featuresBuilder;
        }

        @Override
        public int getFeaturesCount() {
            return featureCount;
        }
    }

    private static class CancerTestProcessor extends TestProcessor {
        private VecBuilder targetBuilder = new VecBuilder();
        private VecBuilder featuresBuilder = new VecBuilder();
        private int featureCount = -1;

        @Override
        public void process(CharSequence arg) {
            final CharSequence[] parts = CharSeqTools.split(arg, ',');
            featureCount = parts.length - 2;
            if (parts[1].charAt(0) == 'M') {
                targetBuilder.append(-1);
            } else {
                targetBuilder.append(1);
            }
            for (int i = 2; i < featureCount + 2; i++) {
                featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
            }
        }

        @Override
        public VecBuilder getTargetBuilder() {
            return targetBuilder;
        }

        @Override
        public VecBuilder getFeaturesBuilder() {
            return featuresBuilder;
        }

        @Override
        public int getFeaturesCount() {
            return featureCount;
        }
    }

    private static class BaseDataReadProcessor extends TestProcessor {
        private VecBuilder targetBuilder = new VecBuilder();
        private VecBuilder featuresBuilder = new VecBuilder();
        private int featureCount = -1;

        @Override
        public void process(CharSequence arg) {
            final CharSequence[] parts = CharSeqTools.split(arg, '\t');
            targetBuilder.append(CharSeqTools.parseDouble(parts[1]));
            if (featureCount < 0)
                featureCount = parts.length - 4;
            else if (featureCount != parts.length - 4)
                throw new RuntimeException("\"Failed to parse line \" + lindex + \":\"");
            for (int i = 4; i < parts.length; i++) {
                featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
            }
        }

        @Override
        public VecBuilder getTargetBuilder() {
            return targetBuilder;
        }

        @Override
        public VecBuilder getFeaturesBuilder() {
            return featuresBuilder;
        }

        @Override
        public int getFeaturesCount() {
            return featureCount;
        }

        @Override
        public void wipe() {
            super.wipe();
            featureCount = -1;
        }
    }

    private static class HIGGSReadProcessor extends TestProcessor {
        private VecBuilder targetBuilder = new VecBuilder();
        private VecBuilder featuresBuilder = new VecBuilder();
        private int featureCount = 28;

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

        @Override
        public VecBuilder getTargetBuilder() {
            return targetBuilder;
        }

        @Override
        public VecBuilder getFeaturesBuilder() {
            return featuresBuilder;
        }

        @Override
        public int getFeaturesCount() {
            return featureCount;
        }
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

    public static void main(String[] args) throws IOException {

        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(ResultFile), "utf-8"))) {

            baseDataBestRMSE(1000, 0.005, SatL2.class);

/*            findIntervalCancerData("L2", L2.class, writer, 0);
            findIntervalCancerData("SteinDifficult", CARTSteinDifficult.class, writer, 0);
            findIntervalCancerData("SteinEasy", CARTSteinEasy.class, writer, 0);
            findIntervalCancerData("LOO", LOOL2.class, writer, 0);
            findIntervalCancerData("SAT", SatL2.class, writer, 0);
            findIntervalCancerData("Reg with 0.4", L2.class, writer, 0.4);
            findIntervalCancerData("SAT + SteinDifficult", CARTSatSteinL2.class, writer, 0); */


/*            findIntervalCTSliceData("L2", L2.class, writer, 0);
            findIntervalCTSliceData("LOO", LOOL2.class, writer, 0);
            findIntervalCTSliceData("SAT", SatL2.class, writer, 0);
            findIntervalCTSliceData("CARTSteinDifficult", CARTSteinDifficult.class, writer, 0);
            findIntervalCTSliceData("CARTSteinEasy", CARTSteinEasy.class, writer, 0);
            findIntervalCTSliceData("Reg with 0.4", L2.class, writer, 0.4); */

/*            findIntervalHiggsData("L2", L2.class, writer, 0);
            findIntervalHiggsData("LOO", LOOL2.class, writer, 0);
            findIntervalHiggsData("SAT", SatL2.class, writer, 0);
//            findIntervalHiggsData("CARTSteinDifficult", CARTSteinDifficult.class, writer, 0);
            findIntervalHiggsData("CARTSteinEasy", CARTSteinEasy.class, writer, 0);
/*            findIntervalHiggsData("Reg with 0.4", CARTL2.class, writer, 0.4); */

/*            findIntervalKSHouseData("L2", L2.class, writer, 0);
            findIntervalKSHouseData("LOO", LOOL2.class, writer, 0);
            findIntervalKSHouseData("SAT", SatL2.class, writer, 0);
            findIntervalKSHouseData("CARTSteinDifficult", CARTSteinDifficult.class, writer, 0);
            findIntervalKSHouseData("CARTSteinEasy", CARTSteinEasy.class, writer, 0);
            findIntervalKSHouseData("Reg with 0.4", L2.class, writer, 0.4);
            findIntervalKSHouseData("Sat + Stein-- + Reg + 0.4", CARTSatSteinL2.class, writer, 0.4); */
        }
        System.exit(0);
    }

    private static void findIntervalCancerData(String msg, Class funcClass, BufferedWriter writer, double regCoeff) {
        int M = 100;
        int iterations = 1000;
        double step = 0.05;
        int depth = 3;

        double best[] = new double[M];

        try {
            TestProcessor processor = new CancerTestProcessor();

            DataML data = readData(processor, LearnCancerFileName, TestCancerFileName);

            DataML cur_data = data;
            for (int i = 0; i < M; i++) {
                System.out.printf("!!!%d", i);
                double auc = findBestAUC(cur_data, iterations, step, funcClass, regCoeff, depth);
                best[i] = auc;
                System.out.printf("\nThe Best AUC for cancerData = %.4fc\n", auc);
                cur_data = bootstrap(data);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            Arrays.sort(best);
            int i = 0;
            while (i < M && best[i] == 0) {
                i++;
            }
            String info = String.format("The interval for cancerData + %s: %d times, %d iterations, %.4f step," +
                            "%d depth [%.7f, %.7f]\n",
                    msg, M, iterations, step, depth,
                    best[i + 5], best[M - 6]);
//            String info = String.format("The value for %d iterations and %.4f step is %.7f",
//                    iterations, step, best[0]);
            System.out.printf(info);
            try {
                writer.write(info);
                writer.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static void findIntervalKSHouseData(String msg, Class funcClass, BufferedWriter writer, double regCoeff) {
        int M = 100;
        int iterations = 2600;
        double step = 0.008;

        double best[] = new double[M];

        try {
            TestProcessor processor = new KSHouseReadProcessor();

            DataML data = readData(processor, LearnKSHouseFileName, TestKSHouseFileName);

            DataML cur_data = data;
            for (int i = 0; i < M; i++) {
                System.out.printf("!!!%d", i);
                double rmse = findBestRMSE(cur_data, iterations, step, funcClass, regCoeff);
                best[i] = rmse;
                System.out.printf("\nThe Best RMSE for ks_House = %.4fc\n", rmse);
                cur_data = bootstrap(data);
            }
        } catch (IOException|NumberFormatException e) {
            e.printStackTrace();
        } finally {
            Arrays.sort(best);
            String info;
            if (5 < M && M - 6 >= 0) {
                info = String.format("The interval for ks_house + %s: %d times, %d iterations, %.4f step, [%.7f, %.7f]\n",
                        msg, M, iterations, step,
                        best[5], best[M - 6]);
            } else {
                info = String.format("The interval for ks_house + %s: %d iterations, %.4f step, %.7f\n",
                        msg, iterations, step,
                        best[0]);
            }
            System.out.printf(info);
            try {
                writer.write(info);
                writer.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static void findIntervalHiggsData(String msg, Class funcClass, BufferedWriter writer, double regCoeff) {
        int M = 100;
        int iterations = 4000;
        double step = 0.3;
        int depth = 6;

        double best[] = new double[M];

        try {
            TestProcessor processor = new HIGGSReadProcessor();

            DataML data = readData(processor, LearnHIGGSFileName, TestHIGGSFileName);

            DataML cur_data = data;
            for (int i = 0; i < M; i++) {
                System.out.printf("!!!%d", i);
                double auc = findBestAUC(cur_data, iterations, step, funcClass, regCoeff, depth);
                best[i] = auc;
                System.out.printf("\nThe Best AUC HIGGS = %.4fc\n", auc);
                cur_data = bootstrap(data);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            Arrays.sort(best);
            String info;
            if (M > 5) {
                info = String.format("The interval for HIGGS + %s: %d times, %d iterations, %.4f step, [%.7f, %.7f]\n",
                        msg, M, iterations, step,
                        best[5], best[M - 6]);
            } else {
                info = String.format("The interval for HIGGS + %s: %d times, %d iterations, %.4f step, %.7f\n",
                        msg, M, iterations, step,
                        best[0]);
            }
            System.out.printf(info);
            try {
                writer.write(info);
                writer.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

/*    private static void bestRSMECTSliceData() {
        try {
            CTSliceReadProcessor processor = new CTSliceReadProcessor();
            DataML data = readData(processor, LearnCTSliceFileName, TestCTSliceFileName);
            double rmse = findBestRMSE(data, 300, 0.1);
            System.out.printf("The best - %.7f", rmse);
        } catch (IOException e) {
            e.printStackTrace();
        }
    } */

    private static void findIntervalCTSliceData(String msg, Class funcClass, BufferedWriter writer, double regCoeff) {
        int M = 100;
        int iterations = 100;
        double step = 0.1;
        double best[] = new double[M];

        try {
            CTSliceReadProcessor processor = new CTSliceReadProcessor();
            DataML data = readData(processor, LearnCTSliceFileName, TestCTSliceFileName);
            DataML cur_data = data;
            for (int i = 0; i < M; i++) {
                double rmse = findBestRMSE(cur_data, iterations, step, funcClass, regCoeff);
                best[i] = rmse;
                System.out.printf("\nThe Best RMSE for CTSlices = %.4fc\n", rmse);
                cur_data = bootstrap(data);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
                Arrays.sort(best);
                int i = 0;
                while (i < M && best[i] == 0) {
                    i++;
                }
                String info = String.format("The interval for CT slices + %s: %d times, %d iterations, %.4f step, [%.7f, %.7f]\n",
                        msg, M, iterations, step,
                        best[i + 5], best[M - 6]);
                System.out.printf(info);
                try {
                    writer.write(info);
                    writer.flush();
                } catch (IOException e) {
                        e.printStackTrace();
                }
        }
    }

    private static double baseDataBestRMSE(int iterations, double step, Class funcClass) throws IOException {
        BaseDataReadProcessor processor = new BaseDataReadProcessor();
        DataML data = readData(processor, LearnBaseDataName, TestBaseDataName);

        return findBestRMSE(data, iterations, step, funcClass, 0);
    }

    private static double findBestAUC(final DataML data, final int iterations, final double step,
                                      final Class func, final double regCoeff, int depth) {
        final GradientBoosting<LLLogit> boosting = new GradientBoosting<>(
                new BootstrapOptimization<>(
                        new com.spbsu.exp.cart.CARTTreeOptimization(
                                GridTools.medianGrid(data.getLearnFeatures(), 32), depth, regCoeff), rnd), func, iterations, step);
        final Action counter = new ProgressHandler() {
            int index = 0;

            @Override
            public void invoke(final Trans partial) {
                System.out.print("\n" + index++);
            }
        };
        final LLLogit learnTarget = new LLLogit(data.getLearnTarget(), data.getLearnFeatures());
        final LLLogit testTarget = new LLLogit(data.getTestTarget(), data.getTestFeatures());

        final AUCCalcer aucCalcerLearn = new AUCCalcer("\tAUC learn:\t", data.getLearnFeatures(),
                data.getLearnTarget());
        final AUCCalcer aucCalcerTest = new AUCCalcer("\tAUC test:\t", data.getTestFeatures(),
                data.getTestTarget());

        boosting.addListener(counter);
        boosting.addListener(aucCalcerLearn);
        boosting.addListener(aucCalcerTest);
        boosting.fit(data.getLearnFeatures(), learnTarget);
        return aucCalcerTest.getMax();
    }

    private static double findBestRMSE(final DataML data, final int iterations, final double step,
                                       final Class funcClass, final double regCoeff) {
        final GradientBoosting<L2> boosting = new GradientBoosting<>(
                new BootstrapOptimization<>(
                        new com.spbsu.exp.cart.CARTTreeOptimization(
                                GridTools.medianGrid(data.getLearnFeatures(), 32), 6, regCoeff), rnd), funcClass, iterations, step);
        final Action counter = new ProgressHandler() {
            int index = 0;

            @Override
            public void invoke(final Trans partial) {
                System.out.print("\n" + index++);
            }
        };
        final L2 learnTarget = new L2(data.getLearnTarget(), data.getLearnFeatures());
        final L2 testTarget = new L2(data.getTestTarget(), data.getTestFeatures());
        final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", data.getLearnFeatures(), learnTarget);
        final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", data.getTestFeatures(), testTarget);

        boosting.addListener(counter);
        boosting.addListener(learnListener);
        boosting.addListener(validateListener);
        boosting.fit(data.getLearnFeatures(), learnTarget);
        return validateListener.getMinRMSE();
    }

    protected static class ScoreCalcer implements ProgressHandler {
        private final String message;
        private final Vec current;
        private final VecDataSet ds;
        private final TargetFunc target;
        private boolean isWrite = true;

        ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target) {
            this(message, ds, target, true);
        }

        ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target, boolean isWrite) {
            this.message = message;
            this.isWrite = isWrite;
            this.ds = ds;
            this.target = target;
            current = new ArrayVec(ds.length());
        }

        double min = 1e10;

        double getMinRMSE() {
            return min;
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

            if (isWrite) System.out.print(message + " " + value);
            min = Math.min(value, min);
            if (isWrite) System.out.print(" best = " + min);
        }
    }


}
