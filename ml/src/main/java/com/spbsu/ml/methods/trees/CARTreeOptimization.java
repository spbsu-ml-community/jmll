package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;

public class CARTreeOptimization extends VecOptimization.Stub<WeightedLoss<? extends L2>> {
  @Override
  public Trans fit(final VecDataSet learn, final WeightedLoss<? extends L2> loss) {
    Optimizer opt = new Optimizer(learn, loss);
    for (int i = 0; i < 8; ++i) {
      opt.step();
    }

    return opt.getTree();
  }

  private class Optimizer {
    private final VecDataSet learn;
    private final WeightedLoss loss;

    private final int[][] order;
    private final int featureCount;
    private final int learnSize;

    private SetManager setManager;

    public Optimizer(final VecDataSet learn, final WeightedLoss loss) {
      this.learn = learn;
      this.loss = loss;
      learnSize = learn.length();
      featureCount = learn.xdim();

      order = new int[featureCount][learnSize];
      for (int i = 0; i < featureCount; ++i) {
        order[i] = learn.order(i);
      }

      setManager = new SetManager(learnSize, loss, learn);
    }

    public boolean step() {
      final int setsCount = setManager.getSetsCount();

      WeightedLoss.Stat[] leftStats = new WeightedLoss.Stat[setsCount];
      WeightedLoss.Stat[] bestRightStats = new WeightedLoss.Stat[setsCount];
      double[] bestScores  = new double[setsCount];
      int[]    bestFeature = new int[setsCount];
      double[] bestSplit   = new double[setsCount];

      for (int i = 0; i < setsCount; ++i) {
        bestScores[i] = setManager.getScore(i);
      }

      for (int f = 0; f < featureCount; ++f) {
        for (int i = 0; i < setsCount; ++i) {
          leftStats[i] = (WeightedLoss.Stat)loss.statsFactory().create();
        }

        for (int i = 0; i < learnSize - 1; ++i) {
          final int curPoint = order[f][i];
          final int curSetIdx = setManager.which(curPoint);
          final WeightedLoss.Stat left = leftStats[curSetIdx].append(curPoint, loss.weight(curPoint));
          final WeightedLoss.Stat right = setManager.getComplement(curSetIdx, left);

          double score = loss.score(left) + loss.score(right);

          double curSplit = learn.data().get(curPoint, f);
          double nextSplit = learn.data().get(order[f][i + 1], f);
          if (score < bestScores[curSetIdx] && /*left.weight > 0.0 && right.weight > 0.0 &&*/ nextSplit != curSplit) {
            bestScores[curSetIdx]  = score;
            bestFeature[curSetIdx] = f;
            bestSplit[curSetIdx]   = curSplit;
            bestRightStats[curSetIdx] = right;
          }
        }
      }

      Condition[] newConds = new Condition[setsCount];
      boolean[] updatedSets = new boolean[setsCount];
      int countSplits = 0;
      for (int i = 0; i < setsCount; ++i) {
        if (bestScores[i] < setManager.getScore(i)) {
          newConds[i] = new Condition(bestSplit[i], bestFeature[i]);
          updatedSets[i] = true;
          countSplits++;
        } else {
          updatedSets[i] = false;
        }
      }

      setManager.update(learn, updatedSets, countSplits, newConds, bestRightStats);

      return countSplits != 0;
    }

    public CARTree getTree() {
      final AbstractNode root = setManager.constructTree();
      return new CARTree(root, learn.xdim());
    }
  }

  private interface AbstractNode {
    double getValue(Vec x);
  }

  private class Leaf implements AbstractNode {
    private final double value;

    public Leaf(final double value) {
      this.value = value;
    }

    public double getValue(Vec x) {
      return value;
    }
  }

  private class Node implements AbstractNode {
    private Condition cond;
    private AbstractNode left;
    private AbstractNode right;

    public Node(final Condition cond) {
      this.cond = cond;
      left = null;
      right = null;
    }

    public double getValue(Vec x) {
      if (cond.satisfied(x)) {
        return left.getValue(x);
      } else {
        return right.getValue(x);
      }
    }

    public final AbstractNode update(boolean isLeft, Condition cond) {
      if (isLeft) {
        left = new Node(cond);
        return left;
      } else {
        right = new Node(cond);
        return right;
      }
    }

    public void setLeaf(boolean isLeft, double value, double weight) {
      if (isLeft) {
        left = new Leaf(value * (weight/(1. + weight)));
      } else {
        right = new Leaf(value * (weight/(1. + weight)));
      }
    }
  }

  private class Set {
    private final WeightedLoss.Stat stat;
    private final Node lastNode;
    private final boolean isLeft;

    public Set(final WeightedLoss.Stat stat, final Node lastNode, final boolean isLeft) {
      this.stat = stat;
      this.lastNode = lastNode;
      this.isLeft = isLeft;
    }

    public Node getLastNode() {
      return lastNode;
    }

    public boolean isLeft() {
      return isLeft;
    }

    public double getScore(final WeightedLoss loss) {
      return loss.score(stat);
    }

    public double getValue(final WeightedLoss loss) {
      return loss.bestIncrement(stat);
    }

    public WeightedLoss.Stat getComplement(final WeightedLoss loss, final WeightedLoss.Stat stat) {
      WeightedLoss.Stat complStat = (WeightedLoss.Stat)loss.statsFactory().create();
      complStat.append(this.stat);
      complStat.remove(stat);
      return complStat;
    }
  }

  private class SetManager {
    private int[] setIdxOfPoint;
    private final int numPoints;
    private final WeightedLoss loss;

    private Set[] sets;
    AbstractNode root;

    public SetManager(final int numPoints, final WeightedLoss loss, final VecDataSet learn) {
      this.loss = loss;
      this.numPoints = numPoints;
      setIdxOfPoint = new int[numPoints];
      for (int i = 0; i < numPoints; i++) {
        setIdxOfPoint[i] = 0;
      }

      root = null;
      WeightedLoss.Stat baseStat = (WeightedLoss.Stat)loss.statsFactory().create();
      for (int i = 0; i < learn.length(); ++i) {
        baseStat.append(i, loss.weight(i));
      }

      sets = new Set[1];
      sets[0] = new Set(baseStat, (Node)root, true);
    }

    public final double getScore(int i) {
      return sets[i].getScore(loss);
    }

    public final WeightedLoss.Stat getComplement(int i, final WeightedLoss.Stat stat) {
      return sets[i].getComplement(loss, stat);
    }

    public int getSetsCount() {
      return sets.length;
    }

    public int which(final int pointIdx) {
      return setIdxOfPoint[pointIdx];
    }

    public void update(final VecDataSet learn,
                       final boolean[] updatedSets, final int count,
                       final Condition[] conditions,
                       final WeightedLoss.Stat[] bestRightStats) {
      final int[] newIdx = newSetIndexes(updatedSets);

      Set[] newSets = new Set[sets.length + count];
      for (int i = 0; i < sets.length; ++i) {
        if (updatedSets[i]) {
          final Node lastNode = sets[i].getLastNode();
          if (lastNode == null) {
            root = new Node(conditions[0]);
            WeightedLoss.Stat leftStat = sets[0].getComplement(loss, bestRightStats[0]);
            newSets[0] = new Set(leftStat, (Node)root, true);
            newSets[1] = new Set(bestRightStats[0], (Node)root, false);
          } else {
            final Node newNode = (Node) lastNode.update(sets[i].isLeft(), conditions[i]);
            WeightedLoss.Stat leftStat = sets[i].getComplement(loss, bestRightStats[i]);
            newSets[newIdx[i]] = new Set(leftStat, newNode, true);
            newSets[newIdx[i] + 1] = new Set(bestRightStats[i], newNode, false);
          }
        } else {
          newSets[newIdx[i]] = sets[i];
        }
      }

      sets = newSets;

      for (int i = 0; i < numPoints; ++i) {
        int setIdx = setIdxOfPoint[i];
        if (updatedSets[setIdx]) {
          int feature = conditions[setIdx].getCondFeature();
          double value = conditions[setIdx].getValue();
          if (learn.data().get(i, feature) <= value) {
            setIdxOfPoint[i] = newIdx[setIdx];
          } else {
            setIdxOfPoint[i] = newIdx[setIdx] + 1;
          }
        } else {
          setIdxOfPoint[i] = newIdx[setIdx];
        }
      }
    }

    private int[] newSetIndexes(final boolean[] updatedSets) {
      int[] newIdx = new int[sets.length];
      int idx = 0;
      for (int i = 0; i < sets.length; ++i) {
        newIdx[i] = idx;
        if (updatedSets[i]) {
          idx++;
        }
        idx++;
      }

      return newIdx;
    }

    public final AbstractNode constructTree() {
      if (root == null) {
        root = new Leaf(sets[0].getValue(loss));
        return root;
      }

      for (int i = 0; i < sets.length; ++i) {
        Node lastNode = sets[i].getLastNode();
        lastNode.setLeaf(sets[i].isLeft(), sets[i].getValue(loss), ((L2.MSEStats)sets[i].stat.inside).weight);
      }
      return root;
    }
  }

  private class Condition {
    private final double value;
    private final int condFeature;

    public Condition(double value, int condFeature) {
      this.value = value;
      this.condFeature = condFeature;
    }

    public boolean satisfied(Vec x) {
      return x.get(condFeature) <= value;
    }

    public int getCondFeature() {
      return condFeature;
    }

    public double getValue() {
      return value;
    }
  }

  private class CARTree extends Func.Stub {
    private final AbstractNode root;
    private final int xdim;

    CARTree(AbstractNode root, int xdim) {
      this.root = root;
      this.xdim = xdim;
    }

    @Override
    public double value(Vec x) {
      return root.getValue(x);
    }

    @Override
    public int dim() {
      return xdim;
    }
  }
}

