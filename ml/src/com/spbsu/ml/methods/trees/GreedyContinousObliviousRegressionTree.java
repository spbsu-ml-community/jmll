package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.methods.GreedyTDRegion;
import com.spbsu.ml.models.ContinousObliviousTree;
import com.spbsu.ml.models.ObliviousTree;
import com.spbsu.ml.optimization.ConvexFunction;
import com.spbsu.ml.optimization.ConvexOptimize;
import com.spbsu.ml.optimization.impl.Nesterov1;

import java.util.List;
import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 21:09
 */
public class GreedyContinousObliviousRegressionTree extends GreedyTDRegion {
    private final int depth;
    private final int numberOfVariables;
    private List<BFGrid.BinaryFeature> features;

    public GreedyContinousObliviousRegressionTree(Random rng, DataSet ds, BFGrid grid, int depth) {
        super(rng, ds, grid, 1. / 3, 0);
        numberOfVariables = (1 << (depth - 1)) * (depth + 1) * (depth + 2);
        this.depth = depth;
    }

    int numOfBoundaries;
    double[] right;

    //Add boundary to resulting matrix
    public void addBoundary(Mx mx, double cond[], double rightPart) {
        for (int i = 0; i < cond.length; i++)
            mx.set(numOfBoundaries, i, cond[i]);
        right[numOfBoundaries] = rightPart;
        numOfBoundaries++;
    }

    //Make 2 dimension index 1
    public int getIndex(int mask, int i, int j) {
        if (i < j) {
            int temp = i;
            i = j;
            j = temp;
        }

        return mask * (depth + 1) * (depth + 2) / 2 + i * (i + 1) / 2 + j;
    }

    //Create boundary of continous between mask and mask neighbour
    public double calcBoundariesFine(double[] value) {
        double sum = 0;
        for (int mask = 0; mask < 1 << depth; mask++)
            for (int _featureNum = 0; _featureNum < depth; _featureNum++) {
                BFGrid.BinaryFeature feature = features.get(_featureNum);
                int featureNum = _featureNum;
                if (((mask >> featureNum) & 1) == 0)
                    continue;

                double C = feature.condition;
                int conterMask = mask ^ (1 << featureNum);
                featureNum++;
                //Equal at 0 point
                {
                    double lambda = 1;
                    double cond = 0;
                    cond += value[getIndex(mask, 0, 0)] * 1;
                    cond += value[getIndex(conterMask, 0, 0)] * -1;
                    cond += value[getIndex(mask, featureNum, featureNum)] * C * C;
                    cond += value[getIndex(conterMask, featureNum, featureNum)] * -C * C;
                    sum += Math.exp(lambda * sqr(cond));
                }
                //Quadratic boundary
                for (int i = 1; i <= depth; i++)
                    for (int j = 1; j <= i; j++)
                        if ((i != featureNum) && (j != featureNum)) {
                            double lambda = 0.1;
                            double cond = 0;
                            cond += value[getIndex(mask, i, j)] * 1;
                            cond += value[getIndex(conterMask, i, j)] * -1;
                            sum += Math.exp(lambda * sqr(cond));
                        }
                //Linear boundary
                for (int i = 1; i <= depth; i++)
                    if (i != featureNum) {
                        double cond = 0;
                        double lambda = 0.4;
                        cond += value[getIndex(mask, 0, i)] * 1;
                        cond += value[getIndex(conterMask, 0, i)] * -1;
                        cond += value[getIndex(mask, featureNum, i)] * C;
                        cond += value[getIndex(conterMask, featureNum, i)] * -C;
                        sum += Math.exp(lambda * sqr(cond));
                    }
            }
        return sum;
    }

    public void createGradientCondition(DataSet ds, int i, int j, Mx mx) {
        double cond[][] = new double[1 << depth][numberOfVariables];
        double R[] = new double[1 << depth];
        for (int k = 0; k < ds.power(); k++) {
            int index = 0;
            //Calculating in which leaf ds[k] lays
            for (BFGrid.BinaryFeature feature : features) {
                index <<= 1;
                if (feature.value(ds.data().row(k)))
                    index++;
            }

            double data[] = new double[depth + 1];
            data[0] = 1;
            for (int s = 0; s < features.size(); s++) {
                data[s + 1] = ds.data().get(k, features.get(s).findex);

            }
            //Gradient condition

            for (int f = 0; f <= depth; f++)
                for (int s = 0; s <= f; s++)
                    cond[index][getIndex(index, f, s)] += data[i] * data[j] * data[f] * data[s];
            R[index] += data[i] * data[j] * ds.target().get(k);
        }

        for (int s = 0; s < 1 << depth; s++)
            addBoundary(mx, cond[s], R[s]);
    }

    //Gauss Method for solving linear equation
    double[] solve(Mx mx, double right[]) {
        int p[] = new int[numberOfVariables];
        int bug = 0, undef = 0;
        for (int i = 0; i < numberOfVariables; i++) p[i] = -1;
        i:
        for (int i = 0; i < numOfBoundaries; i++) {
            for (int j = 0; j < numberOfVariables; j++)
                if (Math.abs(mx.get(i, j)) > 1e-9) {
                    p[j] = i;
                    for (int g = 0; g < numOfBoundaries; g++)
                        if (g != i) {
                            double coef = mx.get(g, j) / mx.get(i, j);
                            for (int k = 0; k < numberOfVariables; k++)
                                mx.set(g, k, mx.get(g, k) - mx.get(i, k) * coef);
                            right[g] -= right[i] * coef;
                        }
                    continue i;
                }
            if (Math.abs(right[i]) > 1e-9) {
                bug++;
                //System.out.println(i);
                //System.out.println(right[i]);
            }


        }

        //System.out.println(mx);
        double ans[] = new double[numberOfVariables];
        for (
                int i = 0;
                i < numberOfVariables; i++)
            if (p[i] == -1)
                undef++;
            else
                ans[i] = right[p[i]] / mx.get(p[i], i);
        System.out.println("\nNot solved expression = " + bug + "Undefined parametrs = " + undef);
        return ans;

    }

    @Override
    public ContinousObliviousTree fit(DataSet learn, Oracle1 loss) {
        return fit(learn, loss, new ArrayVec(learn.power()));
    }

    String outputMatrix(Mx mx, double[] right) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int g = 0; g < numOfBoundaries; g++) {
            for (int i = 0; i < 1 << depth; i++)
                for (int k = 0; k <= depth; k++)
                    for (int j = 0; j <= k; j++)
                        if (Math.abs(mx.get(g, getIndex(i, k, j))) > 1e-9) {
                            stringBuilder.append("c[M = " + Integer.toString(i, 2) + "][" + k + "][" + j + "]*" + mx.get(g, getIndex(i, k, j)) + " +\t");
                        }
            stringBuilder.append("= " + right[g] + "\n");
        }
        return stringBuilder.toString();

    }

    double sqr(double x) {
        return x * x;
    }

    //    boolean firstTime;
    double calculateFine(final double[] value) {
//        long start = System.currentTimeMillis();
        double fine = constCoef;
        for (int i = 0; i < numberOfVariables; i++)
            fine += linearCoef[i] * value[i];
        for (int index = 0; index < 1 << depth; index++)
            for (int i = 0; i < numberOfVariables >> depth; i++)
                for (int j = 0; j < numberOfVariables >> depth; j++)
                    fine += quadraticCoef[index][i][j] * value[index * (numberOfVariables >> depth) + i] * value[index * (numberOfVariables >> depth) + j];

        double bondFine = calcBoundariesFine(value);
/*
        if (Math.abs(oldFine(ds, value) - fine - bondFine) > 1) {
            System.out.println(oldFine(ds, value) + "-" + (fine + bondFine) + "=" + (oldFine(ds, value) - fine - bondFine));
*//*

*/
/*
            for(int i = 0; i < numberOfVariables;i++)
                for(int j =0;j < numberOfVariables;j++)
                    System.out.print(quadraticCoef[i][j] + (j == numberOfVariables -1 ?  "\n" : " "));
*//*
*/
/*

            for (double values : value)
                System.out.println(values);
            System.exit(-1);

*//*

        }

*/
/*
        if(!firstTime){
        System.out.println("Time = " + (System.currentTimeMillis() - start));
        firstTime = true;
        }
*/

        return fine + bondFine;
    }

    double oldFine(DataSet ds, final double[] value) {
        double out[][] = new double[1 << depth][(depth + 1) * (depth + 2) / 2];

        for (int i = 0; i < 1 << depth; i++)
            for (int k = 0; k <= depth; k++)
                for (int j = 0; j <= k; j++)
                    out[i][k * (k + 1) / 2 + j] = value[getIndex(i, k, j)];


        ContinousObliviousTree tree = new ContinousObliviousTree(features, out);
        double fine = 0;
        for (int i = 0; i < ds.power(); i++) {
            fine += sqr(tree.value(ds.data().row(i)) - ds.target().get(i));
            //if(Math.abs(tree.value(ds.data().row(i))) > 1e-9)
            //    System.out.println(tree.value(ds.data().row(i)));
        }
        //fine += calcBoundariesFine(value);
        return fine;
    }

    double quadraticCoef[][][];
    double linearCoef[];
    double constCoef;

    void recalcCoef(DataSet ds) {
        quadraticCoef = new double[1 << depth][numberOfVariables >> depth][numberOfVariables >> depth];
        linearCoef = new double[numberOfVariables];

        for (int i = 0; i < ds.power(); i++) {
            int index = 0;
            for (BFGrid.BinaryFeature feature : features) {
                index <<= 1;
                if (feature.value(ds.data().row(i)))
                    index++;
            }
            double data[] = new double[depth + 1];
            data[0] = 1;
            for (int s = 0; s < features.size(); s++) {
                data[s + 1] = ds.data().get(i, features.get(s).findex);

            }
            double f = ds.target().get(i);
            for (int x = 0; x <= depth; x++)
                for (int y = 0; y <= x; y++) {
                    linearCoef[getIndex(index, x, y)] -= 2 * f * data[x] * data[y];
                }
            for (int x = 0; x <= depth; x++)
                for (int y = 0; y <= x; y++) {
                    for (int x1 = 0; x1 <= depth; x1++)
                        for (int y1 = 0; y1 <= x1; y1++) {
                            quadraticCoef[index][getIndex(0, x, y)][getIndex(0, x1, y1)] += data[x] * data[y] * data[x1] * data[y1];
                        }
                }
            constCoef += sqr(f);
        }


    }

    public class Function implements ConvexFunction {
        DataSet ds;

        @Override
        public int dim() {
            return numberOfVariables;
        }

        @Override
        public double getGlobalConvexParam() {
            return 1;  //To change body of implemented methods use File | Settings | File Templates.
        }

        @Override
        public double getLocalConvexParam(Vec x) {
            return 1;  //To change body of implemented methods use File | Settings | File Templates.
        }

        @Override
        public double getGradLipParam() {
            return 1e5;
        }

        @Override
        public Vec gradient(Vec x) {
            double eps = 1e-9;
            Vec gr = new ArrayVec(numberOfVariables);
            double value[] = x.toArray();
            double curFine = calculateFine(value);
            for (int i = 0; i < numberOfVariables; i++) {
                value[i] += eps;
                double X = calculateFine(value);
                value[i] -= 2 * eps;
                double Y = calculateFine(value);
                gr.set(i, (4 * curFine - 3 * Y - X) / 2 / eps);
                //System.out.println(gr[i]);
                value[i] += eps;

            }

            return gr;  //To change body of implemented methods use File | Settings | File Templates.
        }

        @Override
        public double value(Vec x) {
            return calculateFine(x.toArray());
        }

        Function(DataSet ds) {
            this.ds = ds;
        }
    }

    @Override
    public ContinousObliviousTree fit(DataSet ds, Oracle1 loss, Vec point) {
        features = ((ObliviousTree) new GreedyObliviousRegressionTree(new Random(), ds, grid, depth).fit(ds, loss)).features();
        recalcCoef(ds);
        double out[][] = new double[1 << depth][(depth + 1) * (depth + 2) / 2];

        ConvexFunction function = new Function(ds);
        ConvexOptimize optimize = new Nesterov1(new ArrayVec(numberOfVariables));
        Vec x = optimize.optimize(function, 1);
        double value[] = x.toArray();

        for (int i = 0; i < 1 << depth; i++)
            for (int k = 0; k <= depth; k++)
                for (int j = 0; j <= k; j++)
                    out[i][k * (k + 1) / 2 + j] = value[getIndex(i, k, j)];


        return new ContinousObliviousTree(features, out);
    }


}