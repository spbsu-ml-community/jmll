package com.spbsu.ml.methods.cart;

import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by n_buga on 16.10.16.
 */
public class CARTTreeOptimizationFixError extends VecOptimization.Stub<WeightedLoss<? extends L2>> {

    private List<Leaf> ownerLeafOfData;
    private Vec[] newOrderedFeatures;
    private Vec[] oldOrderedFeatures;
    private VecDataSet newLearn;

    private static Map<Double, Integer> lambdas = new HashMap<>();

    private BiFunction<Func, Vec, Double> retCheckDataError;
    private Function<Vec, Vec> retLearnTarget;

    final ExecutorService executorService = Executors.newFixedThreadPool(4);

    public static void repaint() {
        System.out.println();
        lambdas.keySet().stream().forEach(lmbd -> System.out.printf("%.2f - %d;", lmbd, lambdas.get(lmbd)));
        System.out.println();
        System.out.printf("E = %.3f", lambdas.keySet().stream()
                .flatMap(lmbd -> Stream.generate(() -> lmbd).limit(lambdas.get(lmbd)))
                .collect(Collectors.averagingDouble(d -> d)));
    }

    public CARTTreeOptimizationFixError(VecDataSet learn) {
        newLearn = separate(learn);

        newOrderedFeatures = createOrderFeatures(newLearn);
        oldOrderedFeatures = createOrderFeatures(learn);
    }

    private Vec[] createOrderFeatures(VecDataSet learn) {
        Vec ans[] = new Vec[learn.xdim()];
        final Mx data = learn.data();
        for (int f = 0; f < ans.length; f++) {
            ans[f] = new ArrayVec(learn.length());
            final int[] order = learn.order(f);
            for (int i = 0; i < order.length; i++) {
                ans[f].set(i, data.get(order[i], f));
            }
        }
        return ans;
    }

    private VecDataSet separate(VecDataSet learn) {
        int len = learn.length();
        Random rand = new Random();
        Vec vecArrayNewLearn[] = new Vec[len/2];
        int newLearnIndexes[] = new int[len/2];
        int newCheckIndexes[] = new int[len - len/2];
        int lenLearn = 0;
        int lenCheck = 0;
        for (int i = 0; i < len; i++) {
            int a = rand.nextInt(2);
            if ((lenCheck != len - len/2 && a == 0) || lenLearn == len/2) {
                newCheckIndexes[lenCheck++] = i;
            } else {
                newLearnIndexes[lenLearn] = i;
                vecArrayNewLearn[lenLearn++] = learn.at(i);
            }
        }

        retCheckDataError = (func, target) -> {
            double error = 0;
            for (int newCheckIndex : newCheckIndexes) {
                double funcAns = func.value(learn.at(newCheckIndex));
                double realAns = target.get(newCheckIndex);
                error += Math.abs(funcAns - realAns);
            }
//            error /= newCheckIndexes.length;
            return error;
        };

        retLearnTarget = (Vec target) -> {
            Vec learnTargetArray = new ArrayVec(len/2);
            for (int i = 0; i < newLearnIndexes.length; i++) {
                learnTargetArray.set(i, target.get(newLearnIndexes[i]));
            }
            return learnTargetArray;
        };

        Mx mxNewLearn = new RowsVecArrayMx(vecArrayNewLearn);

        return new VecDataSetImpl(mxNewLearn, null);
    }

    public Trans fit(VecDataSet learn, WeightedLoss loss) {
        final double MAX_COEFF_VALUE = 10;
        final double STEP_SIZE = 0.2;

        double curMinError = Double.POSITIVE_INFINITY;
        double fixCoeff = 0;
        Vec target = loss.target();
        Vec newTarget = retLearnTarget.apply(target);

        for (double j = 0; j < MAX_COEFF_VALUE; j += STEP_SIZE) {
            CARTTree curTree = constructTree(newLearn, newTarget, j, newOrderedFeatures);

            double curError = retCheckDataError.apply(curTree, target);
            if (curError < curMinError) {
                curMinError = curError;
                fixCoeff = j;
            }
        }

        lambdas.put(fixCoeff, lambdas.getOrDefault(fixCoeff, 0) + 1);
        repaint();

        return constructTree(learn, target, fixCoeff, oldOrderedFeatures);
    }

    private CARTTree constructTree(VecDataSet learn, Vec target, double fixCoeff, Vec[] orderedFeatures) {
        List<Leaf> tree = new ArrayList<>();
        Leaf firstLeaf = new Leaf(0);

        ownerLeafOfData = new ArrayList<>(learn.length());
        for (int i = 0; i < learn.length(); i++) {
            ownerLeafOfData.add(firstLeaf);
            firstLeaf.addNewItem(target.get(i));
        }

        firstLeaf.calcError();
        firstLeaf.calcMean();

        tree.add(firstLeaf);

        int count = 0;
        int old_size = tree.size();
        while (count < 7) {
            makeStep(tree, learn, target, fixCoeff, orderedFeatures);
            count++;
            if (old_size == tree.size()) {
                break;
            }
            old_size = tree.size();
        }

        return new CARTTree(tree);
    }

    private double makeStep(List<Leaf> tree, VecDataSet learn, Vec target, double fixCoeff,
                            Vec orderedFeatures[]) { //return maxError along new leaves  ?streams?

        double bestError[] = new double[tree.size()];
        Condition bestCondition[] = new Condition[tree.size()];

        for (int i = 0; i < tree.size(); i++) {
            bestError[i] = Double.POSITIVE_INFINITY;
            bestCondition[i] = new Condition();
        }

        final int dim = learn.xdim();
        final int length = learn.length();

        final Future[] tasks = new Future[dim];

        for (int i = 0; i < dim; i++) { // sort out feature
            final int k = i;
            tasks[i] = executorService.submit(() -> handleFeature(learn, target, fixCoeff, orderedFeatures,
                    bestError, bestCondition,
                    k, tree.size(), length));
        }

        int countLeavesBefore = tree.size();
        Leaf pairLeaf[] = new Leaf[countLeavesBefore];

        try {
            for (int i = 0; i < dim; i++) {
                tasks[i].get();
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < countLeavesBefore; i++) {
            Leaf leaf = tree.get(i);
            if (leaf.getError() <= bestError[i]) { // if new error worse then old
                pairLeaf[i] = leaf;
                continue;
            }

            Leaf newLeaf = new Leaf(leaf, tree.size());
            tree.add(newLeaf);
            pairLeaf[i] = newLeaf;

            leaf.getListFeatures().addFeature(bestCondition[i]);
            newLeaf.getListFeatures().addFeature(new Condition(bestCondition[i]).set(false));

        }

        for (int i = 0; i < learn.length(); i++) {
            Leaf curLeaf = ownerLeafOfData.get(i);
            int leafNumber = curLeaf.getLeafNumber();
            if (!bestCondition[leafNumber].checkFeature(learn.data().row(i))) {
                ownerLeafOfData.set(i, pairLeaf[leafNumber]);
            }
        }

        for (Leaf leaf: tree) {
            leaf.clearStatistic();
        }


        for (int i = 0; i < learn.length(); i++) {
            Leaf curLeaf = ownerLeafOfData.get(i);
            curLeaf.addNewItem(target.get(i));
        }

        double maxErr = 0; //the return value

        for (Leaf leaf: tree) {
            if (leaf.getCount() == 0) {
                continue;
            }
            leaf.calcError();
            leaf.calcMean();
            maxErr = Math.max(maxErr, leaf.getError());
        }

        //0.075, ограничить размер глубины 7.

        //распараллетить нахождение лучше ошибки(например, по value или по feature)
        // дисперсия, исправленная дисперсия, средневыборочная
        // выкинуть фрейм(иконка) - для дебага
        // код грязный, внести апдейт внутрь листа, чтобы до приватных переменных не дошло
        // лист умеет обновляться и делится - вынести в лист.
        // secondPartsum - внутрь лифов.

        // Функция слишком большая - разделить
        // Метод для подсчёта ошибки

        return maxErr;
    }

    private void handleFeature(VecDataSet learn, Vec target, double fixCoeff, Vec[] orderedFeatures,
                               double[] bestError, Condition[] bestCondition,
                               int numFeature, int treeSize, int learnLength) {
        final int[] order = learn.order(numFeature);
        final Vec orderedFeature = orderedFeatures[numFeature];

        int curCount[] = new int[treeSize];
        double partSum[] = new double[treeSize];
        double last[] = new double[treeSize];
        double partSqrtSum[] = new double[treeSize];

        for (int j = 0; j < treeSize; j++) {
            curCount[j] = 0;
            partSum[j] = 0;
            last[j] = 0;
            partSqrtSum[j] = 0;
        }

        for (int j = 0; j < learnLength; j++) { //sort out vector on barrier
            final int curIndex = order[j];                  //check error of this barrier
            final Leaf curLeaf = ownerLeafOfData.get(curIndex);
            final int leafNumber = curLeaf.getLeafNumber();

            final double x_ji = orderedFeature.get(j);

            if (curCount[leafNumber] > 0 && last[leafNumber] < x_ji) { // catch boarder
                final double firstPartSum = partSum[leafNumber];
                final int firstPartCount = curCount[leafNumber];
                final double secondPartSum = curLeaf.getSum() - firstPartSum;
                final int secondPartCount = curLeaf.getCount() - firstPartCount;
                final double firstPartSqrSum = partSqrtSum[leafNumber];
                final double secondPartSqrSum = curLeaf.getSqrSum() - firstPartSqrSum;
                final double errorLeft = score(firstPartSum, firstPartCount, firstPartSqrSum);
                final double errorRight = score(secondPartSum, secondPartCount, secondPartSqrSum);
                final double curError = errorLeft + errorRight + fixCoeff*entropy(curLeaf.getCount(),
                        firstPartCount);
                synchronized (curLeaf) {
                    if (curError < bestError[leafNumber] && curError < curLeaf.getError()) {
                        bestError[leafNumber] = curError;
                        bestCondition[leafNumber].set(numFeature, x_ji, true);
                    }
                }
            }

            double y = target.get(curIndex);

            partSum[leafNumber] += y;
            curCount[leafNumber]++;
            last[leafNumber] = x_ji; //last value of data in this leaf
            partSqrtSum[leafNumber] += y*y;
        }
    }

    private double score(double sum, int count, double sqrSum) {

        double score;
        if (count <= 1) {
            score = Double.POSITIVE_INFINITY;
        } else {
            score = ((sqrSum - (sum * sum) / count)*(count)*(count + 2)/(count*count - 3*count + 1));
        }

        return score;
    }

    private double entropy(int genCount, int leftCount) {
        int rightCount = genCount - leftCount;
        double p1 = (leftCount + 1)*1.0;
        double p2 = (rightCount + 1)*1.0;
        return (- p1*Math.log(p1) - p2*Math.log(p2));
    }
}
