package com.spbsu.ml.data.tools;

import com.spbsu.commons.math.metrics.Metric;
import com.spbsu.commons.math.metrics.impl.CosineDVectorMetric;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.IndexTransVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.ChangedTarget;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.loss.L2;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.io.IOException;
import java.util.Random;

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

  public static int classEntriesCount(Vec target, int classNo) {
    int result = 0;
    for (int i = 0; i < target.dim(); i++) {
      if ((int)target.get(i) == classNo)
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
  public static Vec normalizeTarget(Vec target, TIntList labels) {
    Vec result = new ArrayVec(target.dim());
    for (int i = 0; i < target.dim(); i++) {
      int oldTargetVal = (int) target.get(i);
      int labelPos = labels.indexOf(oldTargetVal);
      if (labelPos == -1) {
        result.set(i, labels.size());
        labels.add(oldTargetVal);
      }
      else
        result.set(i, labelPos);
    }
    return result;
  }

  public static TIntObjectMap<TIntList> splitClassesIdxs(DataSet ds) {
    final TIntObjectMap<TIntList> indexes = new TIntObjectHashMap<TIntList>();
    for (DSIterator iter = ds.iterator(); iter.advance(); ) {
      final int catId = (int) iter.y();
      if (indexes.containsKey(catId)) {
        indexes.get(catId).add(iter.index());
      }
      else {
        final TIntList newClassIdxs = new TIntLinkedList();
        newClassIdxs.add(iter.index());
        indexes.put(catId, newClassIdxs);
      }
    }
    return indexes;
  }

  public static DataSet normalizeClasses(DataSet learn) {
    final DSIterator it = learn.iterator();
    final Vec normalized = new ArrayVec(learn.power());
    while (it.advance()) {
      normalized.set(it.index(), normalizeRelevance(it.y()));
    }
    return new ChangedTarget((DataSetImpl)learn, normalized);
  }

  private static <LocalLoss extends L2> double normalizeRelevance(double y) {
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

  public static Pair<DataSet, DataSet> splitCvMulticlass(DataSet ds, double percentage, Random rnd) {
    final TIntObjectMap<TIntList> indexes = splitClassesIdxs(ds);
    final TIntList learnIdxs = new TIntLinkedList();
    final TIntList testIdxs = new TIntLinkedList();
    for (TIntObjectIterator<TIntList> mapIt = indexes.iterator(); mapIt.hasNext(); ) {
      mapIt.advance();
      for (TIntIterator listIt = mapIt.value().iterator(); listIt.hasNext(); ) {
        if (rnd.nextDouble() < percentage)
          learnIdxs.add(listIt.next());
        else
          testIdxs.add(listIt.next());
      }
    }
    final int[] learnIndicesArr = learnIdxs.toArray();
    final int[] testIndicesArr = testIdxs.toArray();
    return Pair.<DataSet, DataSet>create(
        new DataSetImpl(
            new VecBasedMx(
                ds.xdim(),
                new IndexTransVec(
                    ds.data(),
                    new RowsPermutation(learnIndicesArr, ds.xdim())
                )
            ),
            new IndexTransVec(ds.target(), new ArrayPermutation(learnIndicesArr))
        ),
        new DataSetImpl(
            new VecBasedMx(
                ds.xdim(),
                new IndexTransVec(
                    ds.data(),
                    new RowsPermutation(testIndicesArr, ds.xdim())
                )
            ),
            new IndexTransVec(ds.target(), new ArrayPermutation(testIndicesArr))));
  }

  public static DataSet transformRegressionToMC(DataSet regressionDs, int classCount, TDoubleList borders) throws IOException {
    double[] target = regressionDs.target().toArray();
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

    Vec resultTarget = new ArrayVec(regressionDs.power());
    int targetCursor = 0;
    for (int borderCursor = 0; borderCursor < borders.size(); borderCursor++){
      while (targetCursor < target.length && target[targetCursor] <= borders.get(borderCursor)) {
        resultTarget.set(idxs[targetCursor], borderCursor);
        targetCursor++;
      }
    }
    return new ChangedTarget((DataSetImpl)regressionDs, resultTarget);
  }

  public static DataSet loadRegressionAsMC(String file, int classCount, TDoubleList borders)  throws IOException{
    DataSet ds = DataTools.loadFromFeaturesTxt(file);
    return transformRegressionToMC(ds, classCount, borders);
  }

  public static Mx createSimilarityMatrix(DataSet learn) {
    final TIntObjectMap<TIntList> indexes = splitClassesIdxs(learn);
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
