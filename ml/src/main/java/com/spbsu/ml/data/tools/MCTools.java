package com.spbsu.ml.data.tools;

import java.io.IOException;


import com.spbsu.commons.math.metrics.Metric;
import com.spbsu.commons.math.metrics.impl.CosineDVectorMetric;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.MLLLogit;
import com.spbsu.ml.meta.items.QURLItem;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import static java.lang.Math.max;

/**
 * User: qdeee
 * Date: 04.06.14
 */
public class MCTools {

  public static int countClasses(Vec target) {
    int classesCount = 0;
    for (int i = 0; i < target.dim(); i++) {
      classesCount = max((int)target.get(i) + 1, classesCount);
    }
    return classesCount;
  }

  public static int classEntriesCount(LLLogit target, int classNo) {
    int result = 0;
    for (int i = 0; i < target.dim(); i++) {
      if ((int)target.label(i) == classNo)
        result++;
    }
    return result;
  }

  public static Vec extractClass(Vec target, int classNo) {
    Vec result = new ArrayVec(target.dim());
    for (int i = 0; i < target.dim(); i++) {
      if (target.get(i) == classNo)
        result.set(i, 1.);
      else
        result.set(i, -1.);
    }
    return result;
  }

  public static int[] getClassesLabels(Vec target) {
    TIntArrayList labels = new TIntArrayList();
    for (int i = 0; i < target.dim(); i++) {
      final int label = (int) target.get(i);
      if (!labels.contains(label)) {
        labels.add(label);
      }
    }
    return labels.toArray();
  }

  /**
   * Normalization of multiclass target. Target may contain any labels
   * @param target Target vec with any class labels.
   * @param labels Empty list which will be filled here by classes labels corresponding their order.
   *               Each label will appear once.
   * @return new target vec with classes labels from {0..K}.
   */
  public static MLLLogit normalizeTarget(MLLLogit target, TIntList labels) {
    int[] result = new int[target.dim()];
    for (int i = 0; i < target.dim(); i++) {
      int oldTargetVal = target.label(i);
      int labelPos = labels.indexOf(oldTargetVal);
      if (labelPos == -1) {
        result[i] = labels.size();
        labels.add(oldTargetVal);
      }
      else
        result[i] = labelPos;
    }
    return new MLLLogit(new IntSeq(result));
  }

  public static TIntObjectMap<TIntList> splitClassesIdxs(MLLLogit target) {
    final TIntObjectMap<TIntList> indexes = new TIntObjectHashMap<TIntList>();
    for (int i = 0; i < target.dim(); i++) {
      final int catId = target.label(i);
      if (indexes.containsKey(catId)) {
        indexes.get(catId).add(i);
      }
      else {
        final TIntList newClassIdxs = new TIntLinkedList();
        newClassIdxs.add(i);
        indexes.put(catId, newClassIdxs);
      }
    }
    return indexes;
  }

  private static double normalizeRelevance(double y) {
    if (y <= 0.0)
      return 0.;
//    else if (y < 0.14)
//      return 1.;
//    else if (y < 0.41)
//      return 2.;
//    else if (y < 0.61)
//      return 3.;
//    else
//      return 4.;
    return 1.;
  }


  public static MLLLogit transformRegressionToMC(L2 regression, int classCount, TDoubleList borders) throws IOException {
    double[] target = regression.target.toArray();
    int[] idxs = ArrayTools.sequence(0, target.length);
    ArrayTools.parallelSort(target, idxs);

    if (borders.size() == 0) {
      double min = target[0];
      double max = target[target.length - 1];
      double delta = (max - min) / classCount;
      for (int i = 0; i < classCount; i++) {
        borders.add(delta * (i + 1));
      }
    }

    int[] resultTarget = new int[regression.dim()];
    int targetCursor = 0;
    for (int borderCursor = 0; borderCursor < borders.size(); borderCursor++){
      while (targetCursor < target.length && target[targetCursor] <= borders.get(borderCursor)) {
        resultTarget[idxs[targetCursor]] = borderCursor;
        targetCursor++;
      }
    }
    return new MLLLogit(new IntSeq(resultTarget));
  }

  public static Pair<VecDataSet, MLLLogit> loadRegressionAsMC(String file, int classCount, TDoubleList borders)  throws IOException{
    final Pool<QURLItem> pool = DataTools.loadFromFeaturesTxt(file);
    return Pair.create(pool.vecData(), transformRegressionToMC(pool.target(L2.class), classCount, borders));
  }

  public static Mx createSimilarityMatrix(VecDataSet learn, MLLLogit target) {
    final TIntObjectMap<TIntList> indexes = splitClassesIdxs(target);
    final Metric<Vec> metric = new CosineDVectorMetric();
    final int k = indexes.keys().length;
    final Mx S = new VecBasedMx(k, k);
    for (int i = 0; i < k; i++) {
      final TIntList classIdxsI = indexes.get(i);
      for (int j = i; j < k; j++) {
        final TIntList classIdxsJ = indexes.get(j);
        double value = 0.;
        for (TIntIterator iterI = classIdxsI.iterator(); iterI.hasNext(); ) {
          final int i1 = iterI.next();
          for (TIntIterator iterJ = classIdxsJ.iterator(); iterJ.hasNext(); ) {
            final int i2 = iterJ.next();
            value += 1 - metric.distance(learn.data().row(i1), learn.data().row(i2));
          }
        }
        value /= classIdxsI.size() * classIdxsJ.size();
        S.set(i, j, value);
        S.set(j, i, value);
      }
      System.out.println("class " + i + " is finished!");
    }
    return S;
  }

}
