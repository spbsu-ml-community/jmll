package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.models.ExponentialObliviousTree;
import com.spbsu.ml.optimization.ConvexOptimize;
import com.spbsu.ml.optimization.impl.GradientDescent;

import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 30.11.13
 * Time: 17:48
 * Idea please stop making my code yellow
 */
public class GreedyExponentialObliviousTree extends GreedyContinuesObliviousSoftBondariesRegressionTree {

    private double[][] quadraticMissCoefficient;

    public GreedyExponentialObliviousTree(Random rng, DataSet ds, BFGrid grid, int depth) {
        super(rng, ds, grid, depth);
        //executor = Executors.newFixedThreadPool(4);
    }

    double calcDistanseToRegion(int index, Vec point) {
        double ans = 0;
        for (int i = 0; i < features.size(); i++) {
            if (features.get(i).value(point) != ((index >> i) == 1)) {
                ans += sqr(point.get(features.get(i).findex) - features.get(i).condition);//L2
            }
        }

        return 5 * ans;
    }

    @Override
    void precalculateMissCoefficients(DataSet ds) {
        quadraticMissCoefficient = new double[numberOfVariables][numberOfVariables];
        linearMissCoefficient = new double[numberOfVariables];
        coordinateSum = new double[1 << depth][depth];
        numberOfPointInLeaf = new int[1 << depth];
        for (int i = 0; i < ds.power(); i++) {
            double data[] = new double[depth + 1];
            data[0] = 1;
            for (int s = 0; s < features.size(); s++) {
                data[s + 1] = ds.data().get(i, features.get(s).findex);
            }
            double f = ds.target().get(i);
            for (int index = 0; index < 1 << depth; index++) {
                double weight = Math.exp(-calcDistanseToRegion(index, ds.data().row(i)));
                //System.out.println(weight);
                for (int x = 0; x <= depth; x++)
                    for (int y = 0; y <= x; y++) {
                        linearMissCoefficient[getIndex(index, x, y)] -= 2 * f * data[x] * data[y] * weight;
                    }

            }
            for (int index = 0; index < 1 << depth; index++)
                for (int jindex = 0; jindex < 1 << depth; jindex++) {
                    double weight = Math.exp(-calcDistanseToRegion(index, ds.data().row(i)) - calcDistanseToRegion(jindex, ds.data().row(i)));
                    for (int x = 0; x <= depth; x++)
                        for (int y = 0; y <= x; y++) {
                            for (int x1 = 0; x1 <= depth; x1++)
                                for (int y1 = 0; y1 <= x1; y1++) {
                                    quadraticMissCoefficient[getIndex(index, x, y)][getIndex(jindex, x1, y1)] += data[x] * data[y] * data[x1] * data[y1] * weight;
                                }
                        }
                }
            constMiss += sqr(f);
        }
    }


    @Override
    double[] calculateFineGradient(double[] value) {
        double ans[] = linearMissCoefficient.clone();
        /*for (int i = 0; i < numberOfVariables; i++)
            ans[i] += 2 * regulationCoefficient * value[i];*/

        for (int i = 0; i < numberOfVariables; i++)
            for (int j = 0; j < numberOfVariables; j++)
                ans[i] += 2 * quadraticMissCoefficient[i][j] * value[j];
        return ans;
    }

    @Override
    public ExponentialObliviousTree fit(DataSet ds, L2 loss) {
        features = got.fit(ds, loss).features();
        if (features.size() != depth) {
            System.out.println("Greedy oblivious tree bug");
            System.exit(-1);
        }

        precalculateMissCoefficients(ds);
        precalcContinousConditions();
        System.out.println("Precalc is over");
        double out[][] = new double[1 << depth][(depth + 1) * (depth + 2) / 2];
        //for(int i =0 ;i < linearMissCoefficient.length;i++)
        //    System.out.println(linearMissCoefficient[i]);
        ConvexOptimize optimize = new GradientDescent(new ArrayVec(numberOfVariables));
        //MUST BE CHANGED TO matrix inverse
        Vec x = optimize.optimize(new Function(), 0.5);
        double value[] = x.toArray();
        //calculateFine(value);

        for (int i = 0; i < 1 << depth; i++)
            for (int k = 0; k <= depth; k++)
                for (int j = 0; j <= k; j++)
                    out[i][k * (k + 1) / 2 + j] = value[getIndex(i, k, j)];
        //for(int i =0 ; i < gradLambdas.size();i++)
        //    System.out.println(serializeCondtion(i));
        return new ExponentialObliviousTree(features, out);
    }


}
