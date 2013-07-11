package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.models.ContinousObliviousTree;
import com.spbsu.ml.models.ObliviousTree;

import java.util.List;
import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 21:09
 * To change this template use File | Settings | File Templates.
 */
public class GreedyContinousObliviousTree extends GreedyTDRegion {
    private final int depth;
    private final GreedyObliviousTree nonContinousVersion;

    public GreedyContinousObliviousTree(Random rng, DataSet ds, BFGrid grid, int depth) {
        super(rng, ds, grid, 1. / 3, 0);
        nonContinousVersion = new GreedyObliviousTree(rng, ds, grid, depth);
        this.depth = depth;
    }

    int numOfBondaries;
    double[] right;

    public void createGradientCondition(DataSet ds, int mask, int col, int row, Mx mx) {
        int n = depth + 1;
        double cond[] = new double[(1 << depth) * n * n];
        double rightPart = 0;
        for (int k = 0; k < ds.data().columns(); k++) {
            double X = (col == 0 ? 1 : ds.data().get(k, col - 1)) *
                    (row == 0 ? 1 : ds.data().get(k, row - 1));
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (i == col && j == row)
                        cond[mask * n * n + i * n + j] += X * X;
                    else
                        cond[mask * n * n + i * n + j] += X * (i == 0 ? 1 : ds.data().get(k, i - 1)) *
                                (j == 0 ? 1 : ds.data().get(k, j - 1));
            rightPart += X * ds.target().get(k);     //Contant part of equation
        }
        addBoundary(mx, cond, rightPart);
    }

    public void addBoundary(Mx mx, double cond[], double rightPart) {
        for (int i = 0; i < cond.length; i++)
            mx.set(numOfBondaries, i, cond[i]);
        right[numOfBondaries] = rightPart;
        numOfBondaries++;
    }

    public void createBoundariesCondition(DataSet ds, int mask, BFGrid.BinaryFeature feature, int featureNum, Mx mx) {
        int n = depth + 1;
        if (((mask >> featureNum) & 1) == 0)
            return;
        int conterMask = mask ^ (1 << (featureNum));
        featureNum++;
        double C = feature.condition;
        {
            //equal in the point 0
            double cond[] = new double[(1 << depth) * n * n];
            cond[mask * n * n] = 1;
            cond[conterMask * n * n] = -1;
            cond[mask * n * n + featureNum * n + featureNum] = C * C;
            cond[conterMask * n * n + featureNum * n + featureNum] = -C * C;
            addBoundary(mx, cond, 0);
        }
        //linear condition
        for (int i = 0; i < n; i++)
            if (i != featureNum) {
                double cond[] = new double[(1 << depth) * n * n];
                cond[mask * n * n + i * n] = cond[mask * n * n + i] = 0.5;
                cond[conterMask * n * n + i * n] = cond[conterMask * n * n + i] = -0.5;
                cond[mask * n * n + i * n + featureNum] = cond[mask * n * n + featureNum * n + i] = C / 2;
                cond[conterMask * n * n + i * n + featureNum] =
                        cond[conterMask * n * n + featureNum * n + i] = -C / 2;
                addBoundary(mx, cond, 0);
            }

        //Quadratic condition
        for (int i = 1; i < n; i++)
            for (int j = 1; j < n; j++)
                if (i != featureNum && j != featureNum) {
                    double cond[] = new double[(1 << depth) * n * n];
                    cond[mask * n * n + i * n + i] = C * C;
                    cond[conterMask * n * n + i * n + i] = -C * C;
                    addBoundary(mx, cond, 0);
                }
    }

    @Override
    public ContinousObliviousTree fit(DataSet ds, Oracle1 loss) {
        ObliviousTree x = nonContinousVersion.fit(ds, loss);
        //No continous for a while
        /*Mx mxA = new Mx;
        Vec w = new Vec();
        double w0 = 0;
        QuadraticFunction x= new QuadraticFunction(mxA,w,w0);*/
        List<BFGrid.BinaryFeature> features = x.getFeatures();
        double value[][] = new double[1 << features.size()][(features.size() + 1) * (features.size() + 1)];
        int cnt[] = new int[1 << depth];
        int n = features.size() + 1;
        int condFor = (2 * n * n + features.size() * (n) * (n - 1) + 1);
        int dim = (1 << depth) * condFor / 2;
        if (depth == 1)
            dim = 10;
        else if (depth == 2)
            dim = 52;
        Mx mx = new VecBasedMx(dim, (1 << depth) * n * n);
        right = new double[dim];
        Mx A = new VecBasedMx(dim, (1 << depth) * n * n);
        Mx Q = new VecBasedMx(dim, (1 << depth) * n * n);
        numOfBondaries = 0;
        for (int mask = 0; mask < 1 << depth; mask++) {
            //Optimal for miss
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    createGradientCondition(ds, mask, i, j, mx);
                }
            /*for (int i = 0; i < features.size(); i++)
                createBoundariesCondition(ds, mask, features.get(i), i, mx);*/
        }
        System.out.println(mx);
        System.out.print("Created bounaries");
        System.out.println(numOfBondaries + " " + right.length);
        VecTools.householderLQ(mx, A, Q);
        double solution[] = new double[(1 << depth) * n * n];

        /*System.out.print(Q.toString());
        System.out.println("next");
        System.out.print(mx.toString());
        System.out.println("next");
        System.out.print(A.toString());*/

        for (int i = (1 << depth) * n * n - 1; i >= 0; i--) {
            solution[i] = right[i];
            //System.out.println(right[i] + " " + Q.get(i,i));
            for (int j = i + 1; j < (1 << depth) * n * n; j++)
                solution[i] -= solution[j] * Q.get(i, j);
            if (Q.get(i, i) != 0)
                solution[i] /= Q.get(i, i);
            else
                solution[i] = 0;
        }
        for (int i = 0; i < 1 << depth; i++)
            for (int j = 0; j < n * n; j++) {
                value[i][j] = solution[i * n * n + j];
                //System.out.println(i + " " + j + "=" + value[i][j]);
            }
        //System.out.print(B.toString());
        return new ContinousObliviousTree(features, value);
    }


   /* private class BestBFFinder implements TDoubleDoubleProcedure {
        double score = 0;
        int fold = 0;
        double bestScore = Double.MAX_VALUE;
        int bestFeature = -1;

        int bfIndex = 0;

        final double[] totals;
        final double[] totalWeights;
        final int complexity;

        BestBFFinder(double[] totals, double[] totalWeights, int complexity) {
            this.totals = totals;
            this.totalWeights = totalWeights;
            this.complexity = complexity;
        }

        @Override
        public boolean execute(double weight, double sum) {
            double rightScore = score(this.totalWeights[fold] - weight, totals[fold] - sum, complexity);
            double leftScore = score(weight, sum, complexity);
            score += rightScore + leftScore;
            fold++;
            return true;
        }

        public void advance() {
            if (bestScore > score) {
                bestScore = score;
                bestFeature = bfIndex;
            }
            fold = 0;
            score = 0;
            bfIndex++;
        }

        public int bestSplit() {
            return bestFeature;
        }
    } */
}