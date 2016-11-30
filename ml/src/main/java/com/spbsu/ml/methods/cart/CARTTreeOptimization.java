package com.spbsu.ml.methods.cart;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import jdk.management.resource.internal.FutureWrapper;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Created by n_buga on 16.10.16.
 */
public class CARTTreeOptimization extends VecOptimization.Stub<WeightedLoss<? extends L2>> {

    private List<Leaf> ownerLeafOfData;
    private Vec[] orderedFeatures;

    final ExecutorService executorService = Executors.newFixedThreadPool(4);

    public CARTTreeOptimization(VecDataSet learn) {
        orderedFeatures = new Vec[learn.xdim()];
        final Mx data = learn.data();
        for (int f = 0; f < orderedFeatures.length; f++) {
            orderedFeatures[f] = new ArrayVec(learn.length());
            final int[] order = learn.order(f);
            for (int i = 0; i < order.length; i++) {
                orderedFeatures[f].set(i, data.get(order[i], f));
            }
        }
    }

    public Trans fit(VecDataSet learn, WeightedLoss loss) {
        List<Leaf> tree = new ArrayList<>();
        Leaf firstLeaf = new Leaf(0);

        ownerLeafOfData = new ArrayList<>(learn.length());
        for (int i = 0; i < learn.length(); i++) {
            ownerLeafOfData.add(firstLeaf);
            firstLeaf.addNewItem(loss.target().get(i));
        }

        firstLeaf.calcError();
        firstLeaf.calcValue();

        tree.add(firstLeaf);

        constructTree(tree, learn, loss);

        return new CARTTree(tree);
    }

    private void constructTree(List<Leaf> tree, VecDataSet learn, WeightedLoss loss) {
        int count = 0;
        int old_size = tree.size();
        while (count < 7) {
            makeStep(tree, learn, loss);
            count++;
            if (old_size == tree.size()) {
                break;
            }
            old_size = tree.size();
        }
    }

    private double makeStep(List<Leaf> tree, VecDataSet learn, WeightedLoss loss) { //return maxError along new leaves  ?streams?

        double bestError[] = new double[tree.size()];
        Condition bestCondition[] = new Condition[tree.size()];

        for (int i = 0; i < tree.size(); i++) {
            bestError[i] = Double.POSITIVE_INFINITY;
            bestCondition[i] = new Condition();
        }

        //предподсчитать

        final int dim = learn.xdim();
        final Vec target = loss.target();
        final int length = learn.length();

        final Future[] tasks = new Future[dim];

        for (int i = 0; i < dim; i++) { // sort out feature
            final int k = i;
            tasks[i] = executorService.submit(() -> {
                final int[] order = learn.order(k);
                final Vec orderedFeature = orderedFeatures[k];

                int curCount[] = new int[tree.size()];
                double partSum[] = new double[tree.size()];
                double last[] = new double[tree.size()];
                double partSqrtSum[] = new double[tree.size()];

                for (int j = 0; j < tree.size(); j++) {
                    curCount[j] = 0;
                    partSum[j] = 0;
                    last[j] = 0;
                    partSqrtSum[j] = 0;
                }

                for (int j = 0; j < length; j++) { //sort out vector on barrier
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
                        final double curError = errorLeft + errorRight;
                        synchronized (curLeaf) {
                            if (curError < bestError[leafNumber] && curError < curLeaf.getError()) {
                                bestError[leafNumber] = curError; // и это тоже
                                bestCondition[leafNumber].set(k, x_ji, true);
                            }
                        }
                    }

                    double y = target.get(curIndex);

                    partSum[leafNumber] += y;
                    curCount[leafNumber]++;
                    last[leafNumber] = x_ji;
                    partSqrtSum[leafNumber] += y*y;

                    //last value of data in this leaf
                }
            });
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

/*        executorService.shutdown();
        try {
            executorService.awaitTermination(1, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } */

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
            leaf.calcValue();
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

    private double score(double sum, int count, double sqrSum) {

        double score;
        if (count <= 2) {
            score = Double.POSITIVE_INFINITY;
        } else {
            score = (( - (sum * sum) / count)*(count + 1)/ (count - 1));
        }

        return score;
    }
}
