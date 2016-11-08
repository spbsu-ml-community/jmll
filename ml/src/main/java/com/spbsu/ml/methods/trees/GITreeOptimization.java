package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;

import java.util.ArrayList;
import java.util.List;

import static com.spbsu.commons.math.MathTools.EPSILON;

public class GITreeOptimization extends VecOptimization.Stub<L2> {
  @Override
  public Trans fit(VecDataSet learn, L2 l2Reg) {
    List<Leaf> leaves = new ArrayList<>();

    Leaf grandLeaf = new Leaf();
    leaves.add(grandLeaf);
    int grandLeafIdx = 0;

    L2.MSEStats gLeafMSEStat = new L2.MSEStats(l2Reg.target);
    int[] leafIdxOf = new int[learn.length()];
    int len = learn.length();
    for (int i = 0; i < len; ++i) {
      leafIdxOf[i] = grandLeafIdx;
      gLeafMSEStat.append(i, 1);
    }
    grandLeaf.setMSEStat(gLeafMSEStat);

    List<int[]> order = new ArrayList<>(learn.xdim());
    for (int i = 0; i < learn.xdim(); ++i) {
      order.add(learn.order(i));
    }

    Mx data = learn.data();
    final int n = learn.length();
    final int xdim = learn.xdim();
    int prevLeavesCount;
    int depth = 0;
    do {
      depth++;
      prevLeavesCount = leaves.size();
      L2.MSEStats[] leftMSEStats = new L2.MSEStats[prevLeavesCount];
      double[] leafScore = new double[prevLeavesCount];
      Leaf[]   newLeaves = new Leaf[2 * prevLeavesCount];
      for (int i = 0; i < prevLeavesCount; ++i) {
        newLeaves[2 * i] = null;
        newLeaves[2 * i + 1] = null;
        leafScore[i] = l2Reg.score(leaves.get(i).getMSEStat());
      }

      for (int f = 0; f < xdim; ++f) {
        for (int i = 0; i < prevLeavesCount; ++i) {
          leftMSEStats[i] = l2Reg.statsFactory().create();
        }
        int[] curOrder = order.get(f);
        for (int i = 0; i < n; ++i) {
          int idxCurPoint = curOrder[i];
          int curLeafIdx  = leafIdxOf[idxCurPoint];
          L2.MSEStats cur   = leaves.get(curLeafIdx).getMSEStat();
          L2.MSEStats left  = leftMSEStats[curLeafIdx].append(idxCurPoint, 1);
          L2.MSEStats right = l2Reg.statsFactory().create();
          right.append(cur);
          right.remove(left);

          double splitScore = l2Reg.score(left) + l2Reg.score(right);

          if (splitScore < leafScore[curLeafIdx] &&
                  (l2Reg.score(cur) - splitScore) > EPSILON) {
            leafScore[curLeafIdx] = splitScore;
            double split = data.get(idxCurPoint, f);
            newLeaves[2 * curLeafIdx]     = new Leaf(leaves.get(curLeafIdx), l2Reg);
            newLeaves[2 * curLeafIdx + 1] = new Leaf(leaves.get(curLeafIdx), l2Reg);
            newLeaves[2 * curLeafIdx].addCondition(split, true, f, left);
            newLeaves[2 * curLeafIdx + 1].addCondition(split, false, f, right);
          }
        }
      }

      ArrayList<Leaf> updateLeaves = new ArrayList<>(prevLeavesCount);
      int[] newLeafIdx = new int[2 * prevLeavesCount];
      int leavesCount = 0;
      for (int i = 0; i < prevLeavesCount; ++i) {
        if (newLeaves[2 * i] != null) {
          updateLeaves.add(newLeaves[2 * i]);
          updateLeaves.add(newLeaves[2 * i + 1]);

          newLeafIdx[2 * i] = leavesCount;
          newLeafIdx[2 * i + 1] = leavesCount + 1;

          leavesCount += 2;
        } else {
          updateLeaves.add(leaves.get(i));
          newLeafIdx[2 * i] = leavesCount;
          leavesCount += 1;
        }
      }

      if (leavesCount > prevLeavesCount) {
        leaves = updateLeaves;

        for (int i = 0; i < n; ++i) {
          int curLeaf = leafIdxOf[i];
          if (newLeaves[2 * curLeaf] == null) {
            leafIdxOf[i] = newLeafIdx[2 * curLeaf];
          } else {
            if (newLeaves[2 * curLeaf].checkConditions(data.row(i))) {
              leafIdxOf[i] = newLeafIdx[2 * curLeaf];
            } else {
              leafIdxOf[i] = newLeafIdx[2 * curLeaf + 1];
            }
          }
        }
      }

    } while (/*leaves.size() > prevLeavesCount */ depth < 7);
    System.out.println();
    System.out.print(leaves.size());

    return new GiniIndexTree(leaves, learn.xdim());
  }

  private class Leaf {
    private List<Double>  conditions;
    private List<Boolean> isLefts;
    private List<Integer> condFIndexes;
    private L2.MSEStats   MSEStat;

    public Leaf() {
      conditions   = new ArrayList<>();
      condFIndexes = new ArrayList<>();
      isLefts      = new ArrayList<>();
    }

    public Leaf(Leaf leaf, L2 l2Reg) {
      conditions   = new ArrayList<>();
      condFIndexes = new ArrayList<>();
      isLefts      = new ArrayList<>();

      conditions.addAll(leaf.getConditions());
      isLefts.addAll(leaf.getIsLefts());
      condFIndexes.addAll(leaf.getCondFIndexes());

      MSEStat = new L2.MSEStats(l2Reg.target());
      MSEStat.append(leaf.getMSEStat());
    }

    public List<Double> getConditions() {
      return conditions;
    }

    public List<Boolean> getIsLefts() {
      return isLefts;
    }

    public List<Integer> getCondFIndexes() {
      return condFIndexes;
    }

    public Leaf(Double splitValue, Boolean isLeft, Integer fIndex, L2.MSEStats MSEStat) {
      conditions.add(splitValue);
      isLefts.add(isLeft);
      condFIndexes.add(fIndex);
      this.MSEStat = MSEStat;
    }

    public double getMean() {
      return MSEStat.sum / MSEStat.weight;
    }

    public L2.MSEStats getMSEStat() {
      return MSEStat;
    }

    public void setMSEStat(L2.MSEStats MSEStat) {
      this.MSEStat = MSEStat;
    }

    public void addCondition(Double splitValue, Boolean isLeft, Integer fIndex, L2.MSEStats MSEStat) {
      conditions.add(splitValue);
      isLefts.add(isLeft);
      condFIndexes.add(fIndex);
      this.MSEStat = MSEStat;
    }

    public boolean checkConditions(Vec vec) {
      Boolean result = Boolean.TRUE;
      for (int i = 0; i < conditions.size(); ++i) {
        Double splitValue = conditions.get(i);
        Boolean isLeft = isLefts.get(i);
        Integer fIdx = condFIndexes.get(i);

        if (isLeft) {
          result = result && (vec.at(fIdx) <= splitValue);
        } else {
          result = result && (vec.at(fIdx) > splitValue);
        }
      }

      return result;
    }
  }

  private class GiniIndexTree extends Func.Stub {
    private List<Leaf> leaves;
    private int xdim;

    GiniIndexTree(List<Leaf> leaves, int xdim) {
      this.leaves = leaves;
      this.xdim = xdim;
    }

    @Override
    public double value(Vec x) {
      Leaf curLeaf = leaves.get(0);
      for (Leaf leaf : leaves) {
        if (leaf.checkConditions(x)) {
          curLeaf = leaf;
        }
      }
      return curLeaf.getMean();
    }

    @Override
    public int dim() {
      return xdim;
    }
  }
}

