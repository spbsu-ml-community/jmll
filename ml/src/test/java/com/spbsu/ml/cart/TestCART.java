package com.spbsu.ml.cart;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Processor;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.cart.CARTTreeOptimization;
import com.spbsu.ml.methods.cart.CARTTreeOptimizationFixError;
import gnu.trove.map.hash.TDoubleDoubleHashMap;
import gnu.trove.map.hash.TDoubleIntHashMap;
import org.junit.Test;

import javax.xml.crypto.Data;
import java.io.*;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;
import java.util.zip.GZIPInputStream;

import static com.spbsu.commons.math.MathTools.sqr;
import static java.lang.Math.log;


/**
 * Created by n_buga on 16.10.16.
 */

///home/n_buga/git_repository/jmll/ml/src/test/resources/com/spbsu/ml/allstate_learn.csv

public class TestCART {
    private static final String learnAllstateFileName = "allstate_learn.csv.gz";
    private static final String testAllstateFileName = "allstate_test.csv.gz";
    private static final String learnHIGGSFileName = "HIGGS_learn.csv.gz";
    private static final String testHIGGSFileName = "HIGGS_test.csv.gz";
    private static final String dir = "src/test/resources/com/spbsu/ml";

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
            System.out.print(message + value);
            min = Math.min(value, min);
            System.out.print(" best = " + min);
        }
    }

    private class QualityCalcer implements ProgressHandler {
        Vec residues;
        VecDataSet learn;
        double total = 0;
        int index = 0;

        QualityCalcer(Vec learnTarget, VecDataSet learn) {
            residues = VecTools.copy(learnTarget);
            this.learn = learn;
        }

        @Override
        public void invoke(final Trans partial) {
            if (partial instanceof Ensemble) {
                final Ensemble model = (Ensemble) partial;
                final Trans increment = model.last();

                final TDoubleIntHashMap values = new TDoubleIntHashMap();
                final TDoubleDoubleHashMap dispersionDiff = new TDoubleDoubleHashMap();
                int index = 0;
                final VecDataSet ds = learn;
                for (int i = 0; i < ds.data().rows(); i++) {
                    final double value;
                    if (increment instanceof Ensemble) {
                        value = increment.trans(ds.data().row(i)).get(0);
                    } else {
                        value = ((Func) increment).value(ds.data().row(i));
                    }
                    values.adjustOrPutValue(value, 1, 1);
                    final double ddiff = sqr(residues.get(index)) - sqr(residues.get(index) - value);
                    residues.adjust(index, -model.wlast() * value);
                    dispersionDiff.adjustOrPutValue(value, ddiff, ddiff);
                    index++;
                }
//          double totalDispersion = VecTools.multiply(residues, residues);
                double score = 0;
                for (final double key : values.keys()) {
                    final double regularizer = 1 - 2 * log(2) / log(values.get(key) + 1);
                    score += dispersionDiff.get(key) * regularizer;
                }
//          score /= totalDispersion;
                total += score;
                this.index++;
                System.out.print("\tscore:\t" + score + "\tmean:\t" + (total / this.index));
            }
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
    }

    private static class AllstateReadProcessor extends TestProcessor {

        public AllstateReadProcessor() {
            featuresCount = 15;
        }

        @Override
        public void process(CharSequence arg) {
            final CharSequence[] parts = CharSeqTools.split(arg, ',');
            targetBuilder.append(CharSeqTools.parseDouble(parts[34]));
            featuresBuilder.append(CharSeqTools.parseInt(parts[3]));
            featuresBuilder.append(CharSeqTools.parseInt(parts[4]));
            for (int i = 20; i <= 28; i++) {
                featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
            }
            for (int i = 30; i <= 33; i++) {
                featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
            }
        }
    }

    private class HIGGSReadProcessor extends TestProcessor {

        public HIGGSReadProcessor() {
            featuresCount = 28;
        }

        @Override
        public void process(CharSequence arg) {
            final CharSequence[] parts = CharSeqTools.split(arg, ',');
            targetBuilder.append(CharSeqTools.parseDouble(parts[0]));
            for (int i = 1; i <= 28; i++) {
                featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
            }
        }
    }

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
        CharSeqTools.processLines(inTest, processor);
        data = new VecBasedMx(processor.getFeaturesCount(),
                processor.getFeaturesBuilder().build());
        VecDataSet testFeatures = new VecDataSetImpl(data, null);
        Vec testTarget = processor.getTargetBuilder().build();
        return new DataML(learnFeatures, learnTarget, testFeatures, testTarget);
    }

    public void grBoost(DataML data) {
        FastRandom rng = new FastRandom(0);

        final GradientBoosting<L2> boosting = new GradientBoosting<L2>(
                new BootstrapOptimization<L2>(
                        new CARTTreeOptimization(data.getLearnFeatures()), rng), L2.class,
                10000, 0.002);
        final Action counter = new ProgressHandler() {
            int index = 0;

            @Override
            public void invoke(final Trans partial) {
                System.out.print("\n" + index++);
            }
        };
        final L2 learnTargetL2 = new L2(data.getLearnTarget(), data.getLearnFeatures());
        final L2 testTargetL2 = new L2(data.getTestTarget(), data.getTestFeatures());
        final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", data.getLearnFeatures(), learnTargetL2);
        final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", data.getTestFeatures(), testTargetL2);
        final Action qualityCalcer = new QualityCalcer(data.getLearnTarget(), data.getLearnFeatures());
        boosting.addListener(counter);
        boosting.addListener(learnListener);
        boosting.addListener(validateListener);
        boosting.addListener(qualityCalcer);
        boosting.fit(data.getLearnFeatures(), learnTargetL2);
    }

    @Test
    public void testGRBoostCARTAllstate() throws IOException {
        TestProcessor processor = new AllstateReadProcessor();
        DataML data = readData(processor, learnAllstateFileName, testAllstateFileName);
        grBoost(data);
    }

    @Test
    public void testGRBoostCARTHIGGS() throws IOException {
        TestProcessor processor = new HIGGSReadProcessor();
        DataML data = readData(processor, learnHIGGSFileName, testHIGGSFileName);
        grBoost(data);
    }
}

