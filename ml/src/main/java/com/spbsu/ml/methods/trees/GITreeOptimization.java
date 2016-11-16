package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;

import java.util.ArrayList;
import java.util.List;

public class GITreeOptimization extends VecOptimization.Stub<L2> {
  @Override
  public Trans fit(final VecDataSet learn, final L2 l2) {
    final int featureCount = learn.xdim();
    final int learnSize = learn.length();
    final int[][] order = new int[featureCount][learnSize];
    for (int i = 0; i < featureCount; ++i) {
      order[i] = learn.order(i);
    }

    int[] setIdxOfPoint = new int[learnSize];
    L2.MSEStats baseMSEStat = l2.statsFactory().create();
    for (int i = 0; i < learnSize; ++i) {
      baseMSEStat.append(i, 1);
      setIdxOfPoint[i] = 0;
    }

    OptimizationSetInfo baseOptSetInfo = new OptimizationSetInfo(baseMSEStat, l2, 0);
    List<OptimizationSetInfo> optSetsInfo = new ArrayList<>();
    optSetsInfo.add(baseOptSetInfo);

    List<Leaf> leaves = new ArrayList<>();
    Leaf baseLeaf = new Leaf(new ArrayList<>(), baseOptSetInfo.getMean());
    leaves.add(baseLeaf);

    boolean wasSplitted;
    int depth = 0;
    do {
      depth++;
      Pair<List<OptimizationSetInfo>, Condition[]> result = findBestSubsets(learn, l2, setIdxOfPoint, order, optSetsInfo);
      List<OptimizationSetInfo> newSetsInfo = result.first;
      Condition[] conditions = result.second;

      int newSetsCount = newSetsInfo.size();

      List<Leaf> newLeaves = new ArrayList<>(newSetsCount);
      int[] newSetIdx = new int[2 * optSetsInfo.size()];

      for (int i = 0; i < newSetsCount; ++i) {
        int parentIndex = newSetsInfo.get(i).getParentIndex();
        if (conditions[2 * parentIndex] != null) {
          double leftMean = newSetsInfo.get(i).getMean();
          double rightMean = newSetsInfo.get(i + 1).getMean();

          Leaf parentLeaf = leaves.get(parentIndex);
          Leaf left = parentLeaf.leftSplit(conditions[2 * parentIndex], leftMean);
          Leaf right = parentLeaf.rightSplit(conditions[2 * parentIndex + 1], rightMean);

          newLeaves.add(left);
          newLeaves.add(right);

          newSetsInfo.get(i).setParentIndex(i);
          newSetsInfo.get(i + 1).setParentIndex(i + 1);

          newSetIdx[2 * parentIndex] = i;
          newSetIdx[2 * parentIndex + 1] = i + 1;

          i++;
        } else {
          newLeaves.add(leaves.get(parentIndex));
          newSetsInfo.get(i).setParentIndex(i);
          newSetIdx[2 * parentIndex] = i;
        }
      }

      leaves = newLeaves;
      optSetsInfo = newSetsInfo;

      for (int i = 0; i < learnSize; ++i) {
        int prevSetIdx = setIdxOfPoint[i];
        if (conditions[2 * prevSetIdx] != null) {
          int feature = conditions[2 * prevSetIdx].getCondFeature();
          double value = conditions[2 * prevSetIdx].getValue();
          if (learn.data().get(i, feature) <= value) {
            setIdxOfPoint[i] = newSetIdx[2 * prevSetIdx];
          } else {
            setIdxOfPoint[i] = newSetIdx[2 * prevSetIdx + 1];
          }
        } else {
          setIdxOfPoint[i] = newSetIdx[2 * prevSetIdx];
        }
      }

      wasSplitted = newSetsCount > 0;
    } while (/*wasSplitted*/ depth < 7);

    return new GiniIndexTree(leaves, featureCount);
  }

  private Pair<List<OptimizationSetInfo>, Condition[]> findBestSubsets(final VecDataSet learn, final L2 l2,
                                                                       final int[] setIdxOfPoint, final int[][] order,
                                                                       final List<OptimizationSetInfo> optSetsInfo) {
    final int setsCount = optSetsInfo.size();
    final int maxNewSets = 2 * optSetsInfo.size();
    Condition[] conditions = new Condition[maxNewSets];
    List<OptimizationSetInfo> newSets = new ArrayList<>(maxNewSets);

    final int xdim = learn.xdim();
    final int numSamples = learn.length();

    L2.MSEStats[] leftMSEStats = new L2.MSEStats[setsCount];
    L2.MSEStats[] bestLeftMSEStats = new L2.MSEStats[setsCount];
    double[] bestScores = new double[setsCount];
    int[] bestFeature = new int[setsCount];
    double[] bestSplit = new double[setsCount];
    for (int i = 0; i < setsCount; ++i) {
      bestLeftMSEStats[i] = l2.statsFactory().create();
      bestScores[i] = optSetsInfo.get(i).getScore();
    }

    for (int f = 0; f < xdim; ++f) {
      for (int i = 0; i < setsCount; ++i) {
        leftMSEStats[i] = l2.statsFactory().create();
      }

      for (int i = 0; i < numSamples - 1; ++i) {
        final int curPoint = order[f][i];
        final int curSetIdx = setIdxOfPoint[curPoint];
        final L2.MSEStats left = leftMSEStats[curSetIdx].append(curPoint, 1);
        final L2.MSEStats right = optSetsInfo.get(curSetIdx).getComplement(left, l2);

        double score = l2.score(left) + l2.score(right);

        double curSplit = learn.data().get(curPoint, f);
        double nextSplit = learn.data().get(order[f][i + 1], f);
        if (score < bestScores[curSetIdx] && left.weight > 0.0 && right.weight > 0.0 && nextSplit != curSplit) {
          bestScores[curSetIdx]  = score;
          bestFeature[curSetIdx] = f;
          bestSplit[curSetIdx]   = curSplit;
          bestLeftMSEStats[curSetIdx].sum = left.sum;
          bestLeftMSEStats[curSetIdx].sum2 = left.sum2;
          bestLeftMSEStats[curSetIdx].weight = left.weight;
        }
      }
    }

    for (int i = 0; i < setsCount; ++i) {
      if (bestScores[i] < optSetsInfo.get(i).getScore()) {
        conditions[2 * i] = new Condition(bestSplit[i], bestFeature[i], true);
        conditions[2 * i + 1] = new Condition(bestSplit[i], bestFeature[i], false);
        newSets.add(optSetsInfo.get(i).leftSplit(bestLeftMSEStats[i], l2));
        newSets.add(optSetsInfo.get(i).rightSplit(bestLeftMSEStats[i], l2));
      } else {
        conditions[2 * i] = null;
        conditions[2 * i + 1] = null;
        newSets.add(optSetsInfo.get(i));
      }
    }

    return new Pair<>(newSets, conditions);
  }

  private class OptimizationSetInfo {
    final private double score;
    final private double sum;
    final private double sum2;
    final private double weight;

    private int parentIndex;

    public OptimizationSetInfo(final L2.MSEStats stats, final L2 l2, int parentIndex) {
      sum = stats.sum;
      sum2 = stats.sum2;
      weight = stats.weight;
      score = l2.score(stats);
      this.parentIndex = parentIndex;
    }

    public L2.MSEStats getComplement(final L2.MSEStats stats, final L2 l2) {
      L2.MSEStats compl = l2.statsFactory().create();
      compl.sum = sum;
      compl.sum2 = sum2;
      compl.weight = weight;
      compl.remove(stats);
      return compl;
    }

    public double getMean() {
      return sum / weight;
    }

    public double getScore() {
      return score;
    }

    public int getParentIndex() {
      return parentIndex;
    }

    public void setParentIndex(int parentIndex) {
      this.parentIndex = parentIndex;
    }

    public OptimizationSetInfo leftSplit(final L2.MSEStats leftStat, L2 l2) {
      return new OptimizationSetInfo(leftStat, l2, parentIndex);
    }

    public OptimizationSetInfo rightSplit(final L2.MSEStats leftStat, L2 l2) {
      L2.MSEStats rightStat = l2.statsFactory().create();
      rightStat.sum = sum;
      rightStat.sum2 = sum2;
      rightStat.weight = weight;
      rightStat.remove(leftStat);
      return new OptimizationSetInfo(rightStat, l2, parentIndex);
    }
  }

  private class Condition {
    private final double value;
    private final int condFeature;
    private final boolean isLeft;

    public Condition(double value, int condFeature, boolean isLeft) {
      this.value = value;
      this.condFeature = condFeature;
      this.isLeft = isLeft;
    }

    public boolean satisfied(Vec x) {
      if (isLeft) {
        return x.get(condFeature) <= value;
      } else {
        return x.get(condFeature) > value;
      }
    }

    public int getCondFeature() {
      return condFeature;
    }

    public double getValue() {
      return value;
    }
  }

  private class Leaf {
    private final List<Condition> conditions;
    private final double value;

    public Leaf(List<Condition> conditions, double value) {
      this.conditions = conditions;
      this.value = value;
    }

    public Leaf leftSplit(Condition condition, double value) {
      List<Condition> newConditions = new ArrayList<>(conditions.size() + 1);
      newConditions.addAll(conditions);
      newConditions.add(condition);
      return new Leaf(newConditions, value);
    }

    public Leaf rightSplit(Condition condition, double value) {
      List<Condition> newConditions = new ArrayList<>(conditions.size() + 1);
      newConditions.addAll(conditions);
      newConditions.add(condition);
      return new Leaf(newConditions, value);
    }

    public boolean contains(Vec x) {
      boolean isIn = true;
      for (Condition c: conditions) {
        isIn &= c.satisfied(x);
      }
      return isIn;
    }

    public double getValue() {
      return value;
    }
  }

  private class GiniIndexTree extends Func.Stub {
    private final List<Leaf> leaves;
    private final int xdim;

    GiniIndexTree(List<Leaf> leaves, int xdim) {
      this.leaves = leaves;
      this.xdim = xdim;
    }

    @Override
    public double value(Vec x) {
      Leaf curLeaf = leaves.get(0);
      for (Leaf leaf : leaves) {
        if (leaf.contains(x)) {
          curLeaf = leaf;
        }
      }
      return curLeaf.getValue();
    }

    @Override
    public int dim() {
      return xdim;
    }
  }
}

