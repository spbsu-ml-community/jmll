package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.methods.GreedyTDRegion;
import com.spbsu.ml.models.ContinousObliviousTree;
import com.spbsu.ml.models.ObliviousTree;

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
  private final GreedyObliviousRegressionTree nonContinues;
  private final int numberOfVariables;
  private List<BFGrid.BinaryFeature> features;

  public GreedyContinousObliviousRegressionTree(Random rng, DataSet ds, BFGrid grid, int depth) {
    super(rng, ds, grid, 1. / 3, 0);
    nonContinues = new GreedyObliviousRegressionTree(rng, ds, grid, depth);
    numberOfVariables = (1 << (depth - 1)) * (depth + 1) * (depth + 2);
    numOfBoundaries = 0;

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
  public void createBoundariesCondition(int mask, BFGrid.BinaryFeature feature, int featureNum, Mx mx) {
    if (((mask >> featureNum) & 1) == 0)
      return;

    double C = feature.condition;
    int conterMask = mask ^ (1 << featureNum);
    featureNum++;
    //Equal at 0 point
    {
      double cond[] = new double[numberOfVariables];
      cond[getIndex(mask, 0, 0)] = 1;
      cond[getIndex(conterMask, 0, 0)] = -1;
      cond[getIndex(mask, featureNum, featureNum)] = C * C;
      cond[getIndex(conterMask, featureNum, featureNum)] = -C * C;
      addBoundary(mx, cond, 0);
    }
    //Quadratic boundary
    for (int i = 1; i <= depth; i++)
      for (int j = 1; j <= i; j++)
        if ((i != featureNum) && (j != featureNum)) {
          double cond[] = new double[numberOfVariables];
          cond[getIndex(mask, i, j)] = 1;
          cond[getIndex(conterMask, i, j)] = -1;
          addBoundary(mx, cond, 0);
        }
    //Linear boundary
    for (int i = 1; i <= depth; i++)
      if (i != featureNum) {
        double cond[] = new double[numberOfVariables];
        cond[getIndex(mask, 0, i)] = 1;
        cond[getIndex(conterMask, 0, i)] = -1;
        cond[getIndex(mask, featureNum, i)] = C;
        cond[getIndex(conterMask, featureNum, i)] = -C;
        addBoundary(mx, cond, 0);
      }
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
      for (int s = 0; s < features.size(); s++)
        data[s + 1] = ds.data().get(k, features.get(s).findex);

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
    for (int i = 0; i < mx.columns(); i++) {
      int mi = i;
      for (int j = i + 1; j < mx.rows(); j++)
        if (Math.abs(mx.get(j, i)) > Math.abs(mx.get(mi, i)))
          mi = j;
      for (int j = 0; j < mx.columns(); j++) {
        double temp = mx.get(i, j);
        mx.set(i, j, mx.get(mi, j));
        mx.set(mi, j, temp);
      }
      double temp = right[i];
      right[i] = right[mi];
      right[mi] = temp;
      if (Math.abs(mx.get(i, i)) < 1e-9)
        continue;
      for (int j = 0; j < mx.rows(); j++)
        if (i != j && Math.abs(mx.get(j, i)) > 1e-9) {
          double k = mx.get(j, i) / mx.get(i, i);
          for (int g = 0; g < mx.columns(); g++)
            mx.set(j, g, mx.get(j, g) - k * mx.get(i, g));
          right[j] -= k * right[i];
        }

    }
    double[] ans = new double[mx.columns()];
    int unDef = 0, bug = 0;
    for (int i = 0; i < mx.columns(); i++)
      if ((Math.abs(mx.get(i, i)) < 1e-9)) {

        ans[i] = 0;
        unDef++;
        if (Math.abs(right[i]) > 1e-9)
          bug++;
      } else
        ans[i] = right[i] / mx.get(i, i);
    System.out.println("undef = " + unDef + " bug " + bug);
    return ans;

  }

  @Override
  public ContinousObliviousTree fit(DataSet learn, Oracle1 loss) {
    return fit(learn, loss, new ArrayVec(learn.power()));
  }

  @Override
  public ContinousObliviousTree fit(DataSet ds, Oracle1 loss, Vec point) {
    features = ((ObliviousTree) nonContinues.fit(ds, loss)).features();
    int numberOfConditions = 8 * (1 << (depth - 1)) * ((depth + 1) * (depth + 2) + depth);

    Mx mx = new VecBasedMx(numberOfConditions, numberOfVariables);
    right = new double[numberOfConditions];

    for (int mask = 0; mask < 1 << depth; mask++)
      for (int j = 0; j < depth; j++)
        createBoundariesCondition(mask, features.get(j), j, mx);

    for (int i = 0; i < depth + 1; i++)
      for (int j = 0; j <= i; j++)
        createGradientCondition(ds, j, i, mx);

    double value[] = solve(mx, right);
    double out[][] = new double[1 << depth][(depth + 1) * (depth + 2) / 2];

    for (int i = 0; i < 1 << depth; i++)
      for (int k = 0; k <= depth; k++)
        for (int j = 0; j <= k; j++)
          out[i][k * (k + 1) / 2 + j] = value[getIndex(i, k, j)];


    return new ContinousObliviousTree(features, out);
  }


}