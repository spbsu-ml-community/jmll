package com.spbsu.ml.methods.trees;

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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 21:09
 */
public class GreedyContinuesObliviousSoftBondariesRegressionTree extends GreedyTDRegion {
    private final int depth;
    private final int numberOfVariables;
    private List<BFGrid.BinaryFeature> features;
    private final GreedyObliviousRegressionTree got;
    //private final ExecutorService executor;
    private final int numberOfVariablesByLeaf;
    private final double regulationCoefficient;
    private final boolean softBoundary;
    private final double linearFineLambda, constFineLambda, quadraticFineLambda;
    private final double lipshicParametr;

    public GreedyContinuesObliviousSoftBondariesRegressionTree(Random rng, DataSet ds, BFGrid grid, int depth) {
        super(rng, ds, grid, 1. / 3, 0);
        got = new GreedyObliviousRegressionTree(rng, ds, grid, depth);
        numberOfVariablesByLeaf = (depth + 1) * (depth + 2) / 2;
        numberOfVariables = (1 << depth) * numberOfVariablesByLeaf;
        this.depth = depth;
        regulationCoefficient = 1;
        softBoundary = true;
        lipshicParametr = 1e5;
        linearFineLambda = 0.1;
        constFineLambda = quadraticFineLambda = 1;
        //executor = Executors.newFixedThreadPool(4);
    }

    public GreedyContinuesObliviousSoftBondariesRegressionTree(Random rng, DataSet ds, BFGrid grid, int depth, double regulation,
                                                               boolean softBoundary, double constFineLambda, double linearFineLambda, double quadraticFineLambda, double lipshicParametr) {
        super(rng, ds, grid, 1. / 3, 0);
        this.regulationCoefficient = regulation;
        this.softBoundary = softBoundary;
        this.linearFineLambda = linearFineLambda;
        this.constFineLambda = constFineLambda;
        this.quadraticFineLambda = quadraticFineLambda;
        this.lipshicParametr = lipshicParametr;
        got = new GreedyObliviousRegressionTree(rng, ds, grid, depth);
        numberOfVariablesByLeaf = (depth + 1) * (depth + 2) / 2;
        numberOfVariables = (1 << depth) * numberOfVariablesByLeaf;
        this.depth = depth;
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

    //Calculating fine for condition \sum\limits_{i = 0}^{i == indexes.length - 1} value_{indexes_{i}} * coef_{i} = 0
    public void transformConditionToFineGradient(double lambda, int[] indexes, double[] coef, double[] value, double gr[]) {
        double cond = 0;
        for (int i = 0; i < indexes.length; i++)
            cond += value[indexes[i]] * coef[i];
        if (softBoundary) {
            double lconst = Math.exp(lambda * sqr(cond)) * lambda * 2 * (cond);
            for (int i = 0; i < indexes.length; i++)
                gr[indexes[i]] += lconst * coef[i];
        } else {
            double eps = 0.1;
            for (int i = 0; i < indexes.length; i++) {
                gr[indexes[i]] -= lambda * coef[i] / (cond + eps);
                gr[indexes[i]] += lambda * coef[i] / (eps - cond);
            }

        }
    }

    public double transformConditionToFine(double lambda, int[] indexes, double[] coef, double[] value) {
        double cond = 0;
        for (int i = 0; i < indexes.length; i++)
            cond += value[indexes[i]] * coef[i];
        if (softBoundary) {
            return Math.exp(lambda * sqr(cond)) - 1;
        } else {
            double eps = 0.1;
            return lambda * (Math.log(cond + eps) + Math.log(eps - cond));

        }
    }

    ArrayList<double[]> gradCoef;
    ArrayList<int[]> gradIndex;
    ArrayList<Double> gradLambdas;

    public void calcFlexibleBoundariesFineGradient(double[] value, double gr[]) {
        for (int i = 0; i < gradCoef.size(); i++)
            transformConditionToFineGradient(gradLambdas.get(i), gradIndex.get(i), gradCoef.get(i), value, gr);
    }

    public void precalcContinousConditions() {
        gradCoef = new ArrayList<double[]>();
        gradIndex = new ArrayList<int[]>();
        gradLambdas = new ArrayList<Double>();
        for (int mask = 0; mask < 1 << depth; mask++) {
            for (int _featureNum = 0; _featureNum < depth; _featureNum++) {
                if (((mask >> _featureNum) & 1) == 0) {

                    double C = features.get(_featureNum).condition;
                    int neighbourMask = mask ^ (1 << _featureNum);
/*
                    if((numberOfPointInLeaf[neighbourMask] == 0) && (numberOfPointInLeaf[mask] == 0)) //What the difference what happens in empty leaves
                        continue;
*/
                    int featureNum = _featureNum + 1;
                    //Equal at 0 point
                    {
                        gradLambdas.add(constFineLambda);
                        //Point on a plane, but in the mass center of 2 leafs
                        double[] point = new double[depth + 1];
                        for (int i = 0; i < depth; i++)
                            if ((numberOfPointInLeaf[mask] + numberOfPointInLeaf[neighbourMask]) != 0)
                                point[i + 1] = (coordinateSum[mask][i] + coordinateSum[neighbourMask][i]) / (double) (numberOfPointInLeaf[mask] + numberOfPointInLeaf[neighbourMask]);
                        point[0] = 1;
                        point[featureNum] = C;

                        int cnt = 0;
                        //Condition for equals function in a "point"
                        int index[] = new int[2 * numberOfVariablesByLeaf];
                        double coef[] = new double[2 * numberOfVariablesByLeaf];
                        for (int i = 0; i <= depth; i++)
                            for (int j = 0; j <= i; j++) {
                                index[cnt] = getIndex(mask, i, j);
                                coef[cnt++] = point[i] * point[j];
                                index[cnt] = getIndex(neighbourMask, i, j);
                                coef[cnt++] = -point[i] * point[j];
                            }
                        gradIndex.add(index);
                        gradCoef.add(coef);
                    }
                    //Quadratic boundary
                    //All monoms must have equal coefficient in both leafs
                    for (int i = 1; i <= depth; i++)
                        for (int j = 1; j <= i; j++)
                            if ((i != featureNum) && (j != featureNum)) {
                                gradLambdas.add(quadraticFineLambda);
                                gradIndex.add(new int[]{getIndex(mask, i, j), getIndex(neighbourMask, i, j)});
                                gradCoef.add(new double[]{1, -1});
                            }
                    //Linear boundary
                    for (int i = 1; i <= depth; i++)
                        if (i != featureNum) {
                            gradLambdas.add(linearFineLambda);
                            gradIndex.add(new int[]{getIndex(mask, 0, i), getIndex(neighbourMask, 0, i), getIndex(mask, featureNum, i), getIndex(neighbourMask, featureNum, i)});
                            gradCoef.add(new double[]{1, -1, C, -C});
                        }
                }
            }
        }

    }

    @Override
    public ContinousObliviousTree fit(DataSet learn, Oracle1 loss) {
        return fit(learn, loss, new ArrayVec(learn.power()));
    }

    double sqr(double x) {
        return x * x;
    }

    double[] calculateFineGradient(double[] value) {
        double gr[] = linearMissCoefficient.clone();

        for (int i = 0; i < numberOfVariables; i++)
            gr[i] += 2 * regulationCoefficient * value[i];

        for (int index = 0; index < 1 << depth; index++)
            for (int i = 0, iIndex = index * (numberOfVariablesByLeaf); i < numberOfVariablesByLeaf; i++, iIndex++)
                for (int j = 0, jIndex = index * (numberOfVariablesByLeaf); j < (numberOfVariablesByLeaf); j++, jIndex++)
                    gr[iIndex] += 2 * quadraticMissCoefficient[index][i][j] * value[jIndex];
        calcFlexibleBoundariesFineGradient(value, gr);
        return gr;

    }

    //Not usefull debug code
    double calculateFine(final double[] value) {
        //System.exit(-1);
        double fine = constMiss, regululation = 0, bond = 0;
        for (int i = 0; i < numberOfVariables; i++)
            regululation += regulationCoefficient * sqr(value[i]);
        for (int i = 0; i < numberOfVariables; i++)
            fine += linearMissCoefficient[i] * value[i];
        for (int index = 0; index < 1 << depth; index++)
            for (int i = 0; i < numberOfVariablesByLeaf; i++)
                for (int j = 0; j < numberOfVariablesByLeaf; j++)
                    fine += quadraticMissCoefficient[index][i][j] * value[index * (numberOfVariablesByLeaf) + i] * value[index * (numberOfVariablesByLeaf) + j];
        for (int i = 0; i < gradCoef.size(); i++)
            bond += transformConditionToFine(gradLambdas.get(i), gradIndex.get(i), gradCoef.get(i), value);

        System.out.println("fine =" + fine);
        System.out.println("regul =" + regululation);
        System.out.println("bond = " + bond);
        return fine + regululation + bond;
    }


    double quadraticMissCoefficient[][][];
    double linearMissCoefficient[];
    double constMiss;
    double coordinateSum[][];

    int numberOfPointInLeaf[];

    void precalculateMissCoefficients(DataSet ds) {
        quadraticMissCoefficient = new double[1 << depth][numberOfVariablesByLeaf][numberOfVariablesByLeaf];
        linearMissCoefficient = new double[numberOfVariables];
        coordinateSum = new double[1 << depth][depth];
        numberOfPointInLeaf = new int[1 << depth];
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
            for (int s = 1; s <= depth; s++)
                coordinateSum[index][s - 1] += data[s];
            numberOfPointInLeaf[index]++;
            double f = ds.target().get(i);
            for (int x = 0; x <= depth; x++)
                for (int y = 0; y <= x; y++) {
                    linearMissCoefficient[getIndex(index, x, y)] -= 2 * f * data[x] * data[y];
                }
            for (int x = 0; x <= depth; x++)
                for (int y = 0; y <= x; y++) {
                    for (int x1 = 0; x1 <= depth; x1++)
                        for (int y1 = 0; y1 <= x1; y1++) {
                            quadraticMissCoefficient[index][getIndex(0, x, y)][getIndex(0, x1, y1)] += data[x] * data[y] * data[x1] * data[y1];
                        }
                }
            constMiss += sqr(f);
        }


    }

    public class Function implements ConvexFunction {
        @Override
        public int dim() {
            return numberOfVariables;
        }

        @Override
        public double getGlobalConvexParam() {
            return 1;
        }

        @Override
        public double getLocalConvexParam(Vec x) {
            return 1;
        }

        @Override
        public double getGradLipParam() {
            return lipshicParametr;
        }

        @Override
        public Vec gradient(Vec x) {
            return new ArrayVec(calculateFineGradient(x.toArray()));
        }

        @Override
        public double value(Vec x) {
            return calculateFine(x.toArray());
        }
    }

    @Override
    public ContinousObliviousTree fit(DataSet ds, Oracle1 loss, Vec point) {
        features = ((ObliviousTree) got.fit(ds, loss)).features();
        if (features.size() != depth) {
            System.out.println("Greedy oblivious tree bug");
            System.exit(-1);
        }

        precalculateMissCoefficients(ds);
        precalcContinousConditions();
        double out[][] = new double[1 << depth][(depth + 1) * (depth + 2) / 2];
        //for(int i =0 ;i < linearMissCoefficient.length;i++)
        //    System.out.println(linearMissCoefficient[i]);
        ConvexOptimize optimize = new Nesterov1(new ArrayVec(numberOfVariables));
        Vec x = optimize.optimize(new Function(), 0.5);
        double value[] = x.toArray();
        //calculateFine(value);

        for (int i = 0; i < 1 << depth; i++)
            for (int k = 0; k <= depth; k++)
                for (int j = 0; j <= k; j++)
                    out[i][k * (k + 1) / 2 + j] = value[getIndex(i, k, j)];

        return new ContinousObliviousTree(features, out);
    }


}