package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
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
    Integer grandLeafIdx = 0;

    int[] leafIdxOf = new int[learn.length()];
    int len = learn.length();
    for (int i = 0; i < len; ++i) {
      leafIdxOf[i] = grandLeafIdx;
    }

    grandLeaf.setMSEStat(new L2.MSEStats(l2Reg.target));

    List<int[]> order = new ArrayList<>(learn.xdim());
    for (int i = 0; i < learn.xdim(); ++i) {
      order.add(learn.order(i));
    }

    Mx data = learn.data();
    int leavesCount;
    int n = learn.length();
    int xdim = learn.xdim();
    do {
      leavesCount = leaves.size();
      L2.MSEStats[][] leftMSEStats = new L2.MSEStats[xdim][leavesCount];
      double[] impDeltaLeaf = new double[leavesCount];
      Leaf[]   newLeaves    = new Leaf[2 * leavesCount];

      for (int f = 0; f < xdim; ++f) {
        for (int i = 0; i < leaves.size(); ++i) {
          leftMSEStats[f][i] = l2Reg.statsFactory().create();
          impDeltaLeaf[i]    = l2Reg.value(leaves.get(i).getMSEStat());
          newLeaves[2 * i] = null;
          newLeaves[2 * i + 1] = null;
        }
        for (int i = 0; i < n; ++i) {
          int idxCurPoint = order.get(f)[i];
          int curLeafIdx  = leafIdxOf[idxCurPoint];
          L2.MSEStats cur   = leaves.get(curLeafIdx).getMSEStat();
          L2.MSEStats left  = leftMSEStats[f][i].append(idxCurPoint, 1);
          L2.MSEStats right = cur.remove(left);

          double impurityDelta = l2Reg.value(cur) - left.sum / left.weight * l2Reg.value(left)
                  - right.sum / right.weight * l2Reg.value(right);

          if (impurityDelta < impDeltaLeaf[curLeafIdx]) {
            double split = data.get(idxCurPoint, f);
            newLeaves[2 * curLeafIdx]     = new Leaf(leaves.get(curLeafIdx));
            newLeaves[2 * curLeafIdx + 1] = new Leaf(leaves.get(curLeafIdx));
            newLeaves[2 * curLeafIdx].addCondition(split, true, f, left);
            newLeaves[2 * curLeafIdx + 1].addCondition(split, false, f, right);
          }

          leftMSEStats[f][curLeafIdx].append(idxCurPoint, 1);
        }
      }

      int[] newLeafIdx = new int [2 * leavesCount];
      int leavesSize = leaves.size();
      for (int i = 0; i < leavesCount; ++i) {
        if (newLeaves[2 * i] != null) {
          if (impDeltaLeaf[i] > EPSILON) {
            leaves.remove(i);
            leaves.add(newLeaves[2 * i]);
            leaves.add(newLeaves[2 * i + 1]);
            newLeafIdx[2 * i] = leavesSize;
            newLeafIdx[2 * i + 1] = leavesSize + 1;
            leavesSize += 1;
          }
        }
      }

      for (int i = 0; i < n; ++i) {
        int curLeaf = leafIdxOf[i];
        if (newLeaves[2 * curLeaf] != null) {
          if (newLeaves[2 * curLeaf].checkConditions(data.row(i))) {
            leafIdxOf[i] = newLeafIdx[2 * i];
          } else {
            leafIdxOf[i] = newLeafIdx[2 * i + 1];
          }
        }
      }

    } while (leaves.size() > leavesCount);

    return new GiniIndexTree(leaves, learn.xdim(), l2Reg.ydim());
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

    public Leaf(Leaf leaf) {
      this.conditions   = leaf.getConditions();
      this.isLefts      = leaf.getIsLefts();
      this.condFIndexes = leaf.getCondFIndexes();
      this.MSEStat      = leaf.getMSEStat();
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

    public Vec getMean() {
      return new ArrayVec(MSEStat.sum / MSEStat.weight);
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
          result = result && (vec.at(fIdx) < splitValue);
        } else {
          result = result && (vec.at(fIdx) >= splitValue);
        }
      }

      return result;
    }
  }

  private class GiniIndexTree extends Trans.Stub {
    private List<Leaf> leaves;
    private int xdim;
    private int ydim;

    GiniIndexTree(List<Leaf> leaves, int xdim, int ydim) {
      this.leaves = leaves;
      this.xdim = xdim;
      this.ydim = ydim;
    }

    @Override
    public int xdim() {
      return xdim;
    }

    @Override
    public int ydim() {
      return ydim;
    }

    public Vec trans(final Vec vec) {
      Leaf curLeaf = leaves.get(0);
      for (Leaf leaf : leaves) {
        if (leaf.checkConditions(vec)) {
           curLeaf = leaf;
        }
      }
      return curLeaf.getMean();
    }
  }
}

