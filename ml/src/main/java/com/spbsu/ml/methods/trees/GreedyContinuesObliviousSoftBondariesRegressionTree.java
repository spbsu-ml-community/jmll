package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.greedyRegion.GreedyTDRegion;
import com.spbsu.ml.models.ContinousObliviousTree;
import com.spbsu.ml.optimization.FuncConvex;
import com.spbsu.ml.optimization.Optimize;
import com.spbsu.ml.optimization.impl.Nesterov1;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 21:09
 */
public class GreedyContinuesObliviousSoftBondariesRegressionTree extends GreedyTDRegion {
  protected final int depth;
  protected final int numberOfVariables;
  protected List<BFGrid.BinaryFeature> features;
  protected final GreedyObliviousTree got;
  private ExecutorService executor;
  private final int numberOfVariablesByLeaf;
  protected final double regulationCoefficient;
  private final boolean softBoundary;
  private final double linearFineLambda, constFineLambda, quadraticFineLambda;
  private final double lipshicParametr;

  public GreedyContinuesObliviousSoftBondariesRegressionTree(final Random rng, final DataSet ds, final BFGrid grid, final int depth) {
    super(grid);
    got = new GreedyObliviousTree(grid, depth);
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

  public GreedyContinuesObliviousSoftBondariesRegressionTree(final Random rng, final DataSet ds, final BFGrid grid, final int depth, final double regulation,
                                                             final boolean softBoundary, final double constFineLambda, final double linearFineLambda, final double quadraticFineLambda, final double lipshicParametr) {
    super(grid);
    this.regulationCoefficient = regulation;
    this.softBoundary = softBoundary;
    this.linearFineLambda = linearFineLambda;
    this.constFineLambda = constFineLambda;
    this.quadraticFineLambda = quadraticFineLambda;
    this.lipshicParametr = lipshicParametr;
    got = new GreedyObliviousTree(grid, depth);
    numberOfVariablesByLeaf = (depth + 1) * (depth + 2) / 2;
    numberOfVariables = (1 << depth) * numberOfVariablesByLeaf;
    this.depth = depth;
    //executor = Executors.newFixedThreadPool(4);
  }

  //Make 2 dimension index 1
  public int getIndex(final int mask, int i, int j) {
    if (i < j) {
      final int temp = i;
      i = j;
      j = temp;
    }

    return mask * (depth + 1) * (depth + 2) / 2 + i * (i + 1) / 2 + j;
  }

  AtomicReferenceArray<Double> gr;

  void atomicIncrement(final int i, final double x) {
    double old = gr.get(i);
    while (!gr.weakCompareAndSet(i, old, old + x))
      old = gr.get(i);
  }

  //Calculating fine for condition \sum\limits_{i = 0}^{i == indexes.length - 1} value_{indexes_{i}} * coef_{i} = 0
  public void transformConditionToFineGradient(final double lambda, final int[] indexes, final double[] coef, final double[] value) {
    double cond = 0;
    for (int i = 0; i < indexes.length; i++)
      cond += value[indexes[i]] * coef[i];
    final double eps = 0.1;
    for (int i = 0; i < indexes.length; i++) {
      atomicIncrement(indexes[i], lambda * coef[i] / Math.pow(cond + eps, 3));
      atomicIncrement(indexes[i], -lambda * coef[i] / Math.pow(eps - cond, 3));
    }
  }


  public double transformConditionToFine(final double lambda, final int[] indexes, final double[] coef, final double[] value) {
    double cond = 0;
    for (int i = 0; i < indexes.length; i++)
      cond += value[indexes[i]] * coef[i];
    if (softBoundary) {
      return Math.exp(lambda * sqr(cond)) - 1;
    } else {
      final double eps = 0.1;
      return lambda * (Math.log(cond + eps) + Math.log(eps - cond));

    }
  }

  ArrayList<double[]> gradCoef;
  ArrayList<int[]> gradIndex;
  ArrayList<Double> gradLambdas;

  class myThread implements Runnable {
    final int i;
    final double[] value;

    myThread(final int contditionNum, final double[] value) {
      this.i = contditionNum;
      this.value = value;
    }

    @Override
    public void run() {
      transformConditionToFineGradient(gradLambdas.get(i), gradIndex.get(i), gradCoef.get(i), value);
    }
  }

  public void addInPointEqualCondition(final double[] point, final int mask, final int neighbourMask) {
    gradLambdas.add(constFineLambda);
    //Point on a plane, but in the mass center of 2 leafs
    int cnt = 0;
    //Condition for equals function in a "point"
    final int[] index = new int[2 * numberOfVariablesByLeaf];
    final double[] coef = new double[2 * numberOfVariablesByLeaf];
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

  public void precalcContinousConditions() {
    gradCoef = new ArrayList<double[]>();
    gradIndex = new ArrayList<int[]>();
    gradLambdas = new ArrayList<Double>();
    for (int mask = 0; mask < 1 << depth; mask++) {
      for (int _featureNum = 0; _featureNum < depth; _featureNum++) {
        if (((mask >> _featureNum) & 1) == 0) {

          final double C = features.get(_featureNum).condition;
          final int neighbourMask = mask ^ (1 << _featureNum);
/*
                    if((numberOfPointInLeaf[neighbourMask] == 0) && (numberOfPointInLeaf[mask] == 0)) //What the difference what happens in empty leaves
                        continue;
*/
          final int featureNum = _featureNum + 1;
          //Equals at 0 points
          {

            final double[] point = new double[depth + 1];
            for (int i = 0; i < depth; i++)
              if ((numberOfPointInLeaf[mask] + numberOfPointInLeaf[neighbourMask]) != 0)
                point[i + 1] = (coordinateSum[mask][i] + coordinateSum[neighbourMask][i]) / (double) (numberOfPointInLeaf[mask] + numberOfPointInLeaf[neighbourMask]);
            point[0] = 1;
            point[featureNum] = C;
            addInPointEqualCondition(point, mask, neighbourMask);
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

  double sqr(final double x) {
    return x * x;
  }

  String serializeCondtion(final int i) {
    final StringBuilder sb = new StringBuilder();
    sb.append("\\lambda = ").append(gradLambdas.get(i));
    sb.append(" Condition - ");
    for (int j = 0; j < gradIndex.get(i).length; j++)
      sb.append("c_{").append(gradIndex.get(i)[j]).append("} * ").append(gradCoef.get(i)[j]).append(" + ");
    sb.append("= 0");
    return sb.toString();

  }

  double[] calculateFineGradient(final double[] value) {
    final double[] ans = linearMissCoefficient.clone();
    for (int i = 0; i < numberOfVariables; i++)
      ans[i] += 2 * regulationCoefficient * value[i];

    //Optimize place can be optimized 2 time because matrix is semmetric
    for (int index = 0; index < 1 << depth; index++)
      for (int i = 0, iIndex = index * (numberOfVariablesByLeaf); i < numberOfVariablesByLeaf; i++, iIndex++)
        for (int j = 0, jIndex = index * (numberOfVariablesByLeaf); j < (numberOfVariablesByLeaf); j++, jIndex++)
          ans[iIndex] += 2 * quadraticMissCoefficient[index][i][j] * value[i];
    executor = Executors.newFixedThreadPool(2);
    gr = new AtomicReferenceArray<Double>(numberOfVariables);
    for (int i = 0; i < numberOfVariables; i++)
      gr.set(i, 0.);
    for (int i = 0; i < gradCoef.size(); i++)
      executor.submit(new myThread(i, value));
    executor.shutdown();

    try {
      executor.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
    } catch (InterruptedException e) {
      e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
    }

    for (int i = 0; i < numberOfVariables; i++)
      ans[i] += gr.get(i);

    return ans;

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

  void precalculateMissCoefficients(final DataSet ds, final L2 loss) {
    quadraticMissCoefficient = new double[1 << depth][numberOfVariablesByLeaf][numberOfVariablesByLeaf];
    linearMissCoefficient = new double[numberOfVariables];
    coordinateSum = new double[1 << depth][depth];
    numberOfPointInLeaf = new int[1 << depth];
    for (int i = 0; i < ds.length(); i++) {
      int index = 0;
      for (final BFGrid.BinaryFeature feature : features) {
        index <<= 1;
        if (feature.value(((VecDataSet) ds).data().row(i)))
          index++;
      }
      final double[] data = new double[depth + 1];
      data[0] = 1;
      for (int s = 0; s < features.size(); s++) {
        data[s + 1] = ((VecDataSet) ds).data().get(i, features.get(s).findex);
      }
      for (int s = 1; s <= depth; s++)
        coordinateSum[index][s - 1] += data[s];
      numberOfPointInLeaf[index]++;
      final double f = loss.target.get(i);
      for (int x = 0; x <= depth; x++)
        for (int y = 0; y <= x; y++) {
          linearMissCoefficient[getIndex(index, x, y)] -= 2 * f * data[x] * data[y];
        }
      //Optimize place can be optimized 2 time because matrix is semmetric
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

  public class Function extends FuncConvex.Stub {
    @Override
    public int dim() {
      return numberOfVariables;
    }

    @Override
    public double getGlobalConvexParam() {
      return 1;
    }

    @Override
    public double getGradLipParam() {
      return lipshicParametr;
    }

    @Override
    public double value(final Vec x) {
      return calculateFine(x.toArray());
    }

    @Override
    public Vec gradient(final Vec x) {
      return new ArrayVec(calculateFineGradient(x.toArray()));
    }
  }

  public ContinousObliviousTree fit(final VecDataSet ds, final L2 loss) {
    features = got.fit(ds, loss).features();
    if (features.size() != depth) {
      System.out.println("Greedy oblivious tree bug");
      System.exit(-1);
    }

    precalculateMissCoefficients(ds, loss);
    precalcContinousConditions();
    final double[][] out = new double[1 << depth][(depth + 1) * (depth + 2) / 2];
    //for(int i =0 ;i < linearMissCoefficient.length;i++)
    //    System.out.println(linearMissCoefficient[i]);
    final Optimize<FuncConvex> optimize = new Nesterov1(new ArrayVec(numberOfVariables), 0.5);
    final Vec x = optimize.optimize(new Function());
    final double[] value = x.toArray();
    //calculateFine(value);

    for (int i = 0; i < 1 << depth; i++)
      for (int k = 0; k <= depth; k++)
        for (int j = 0; j <= k; j++)
          out[i][k * (k + 1) / 2 + j] = value[getIndex(i, k, j)];
    //for(int i =0 ; i < gradLambdas.size();i++)
    //    System.out.println(serializeCondtion(i));
    return new ContinousObliviousTree(features, out);
  }


}