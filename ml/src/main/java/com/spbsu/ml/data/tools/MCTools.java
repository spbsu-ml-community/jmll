package com.spbsu.ml.data.tools;

import com.spbsu.commons.math.metrics.Metric;
import com.spbsu.commons.math.metrics.impl.CosineDVectorMetric;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.text.StringUtils;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.multiclass.util.ConfusionMatrix;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.models.MCModel;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.io.IOException;
import java.util.Arrays;

import static java.lang.Math.max;

/**
 * User: qdeee
 * Date: 04.06.14
 */
public class MCTools {
  public static int countClasses(IntSeq target) {
    int classesCount = 0;
    for (int i = 0; i < target.length(); i++) {
      classesCount = max(target.at(i) + 1, classesCount);
    }
    return classesCount;
  }

  public static int classEntriesCount(IntSeq target, int classNo) {
    int result = 0;
    for (int i = 0; i < target.length(); i++) {
      if (target.at(i) == classNo)
        result++;
    }
    return result;
  }

  public static Vec extractClassForBinary(IntSeq target, int classNo) {
    final Vec result = new ArrayVec(target.length());
    for (int i = 0; i < target.length(); i++)
      result.set(i, (target.at(i) == classNo) ? 1. : -1.);
    return result;
  }

  /**
   *
   * @param target MC target with any classes labels
   * @return classes labels corresponding their order (uniq)
   */
  public static int[] getClassesLabels(IntSeq target) {
    final TIntList labels = new TIntArrayList();
    for (int i = 0; i < target.length(); i++) {
      final int label = target.at(i);
      if (!labels.contains(label)) {
        labels.add(label);
      }
    }
    return labels.toArray();
  }

  public static int[] getClassLabels(Vec target) {
    final TIntList labels = new TIntArrayList();
    for (int i = 0; i < target.length(); i++) {
      final int label = target.at(i).intValue();
      if (!labels.contains(label)) {
        labels.add(label);
      }
    }
    return labels.toArray();
  }

  /**
   * Normalization of multiclass target. Target may contain any labels. Notice that error class (-1) will be mapped to the class K.
   * Example: if target contains {10, 10, 6, 4, -1, -1} then result is {2, 2, 1, 0, 3, 3} and map will be filled {(4->0), (6->1), (10->2), (-1->3)}
   * @param target Target vec with any class labels.
   * @param labelsMap Empty map which will be filled here by pairs (label, normalizedLabel).
   * @return new target with classes labels from {0..K}.
   */
  public static IntSeq normalizeTarget(final IntSeq target, final TIntIntMap labelsMap) {
    for (int i = 0; i < target.length(); i++) {
      labelsMap.putIfAbsent(target.arr[i], 0);
    }
    labelsMap.remove(-1);

    final int[] labels = labelsMap.keys();
    Arrays.sort(labels);
    for (int i = 0; i < labels.length; i++) {
      labelsMap.put(labels[i], i);
    }
    labelsMap.put(-1, labels.length);

    final int[] newTarget = new int[target.length()];
    for (int i = 0; i < target.length(); i++) {
      newTarget[i] = labelsMap.get(target.arr[i]);
    }
    return new IntSeq(newTarget);
  }

  public static TIntObjectMap<TIntList> splitClassesIdxs(IntSeq target) {
    final TIntObjectMap<TIntList> indexes = new TIntObjectHashMap<TIntList>();
    for (int i = 0; i < target.length(); i++) {
      final int label = target.at(i);
      if (indexes.containsKey(label)) {
        indexes.get(label).add(i);
      }
      else {
        final TIntList newClassIdxs = new TIntLinkedList();
        newClassIdxs.add(i);
        indexes.put(label, newClassIdxs);
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


  public static IntSeq transformRegressionToMC(Vec regressionTarget, int classCount, TDoubleList borders) throws IOException {
    final double[] target = regressionTarget.toArray();
    final int[] idxs = ArrayTools.sequence(0, target.length);
    ArrayTools.parallelSort(target, idxs);

    if (borders.size() == 0) {
      final double min = target[0];
      final double max = target[target.length - 1];
      final double delta = (max - min) / classCount;
      for (int i = 0; i < classCount; i++) {
        borders.add(delta * (i + 1));
      }
    }

    final int[] resultTarget = new int[target.length];
    int targetCursor = 0;
    for (int borderCursor = 0; borderCursor < borders.size(); borderCursor++){
      while (targetCursor < target.length && target[targetCursor] <= borders.get(borderCursor)) {
        resultTarget[idxs[targetCursor]] = borderCursor;
        targetCursor++;
      }
    }
    return new IntSeq(resultTarget);
  }

  public static Pair<VecDataSet, IntSeq> loadRegressionAsMC(String file, int classCount, TDoubleList borders)  throws IOException{
    final Pool<QURLItem> pool = DataTools.loadFromFeaturesTxt(file);
    return Pair.create(pool.vecData(), transformRegressionToMC(pool.target(L2.class).target, classCount, borders));
  }

  public static Mx createSimilarityMatrix(VecDataSet learn, IntSeq target) {
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

  public static String evalModel(final MCModel model, final Pool<?> ds, final String prefixComment, final boolean oneLine) {
    final Vec predict = model.bestClassAll(ds.vecData().data());
    final TIntIntMap labelsMap = new TIntIntHashMap();
    final ConfusionMatrix confusionMatrix = new ConfusionMatrix(
        normalizeTarget(ds.target(BlockwiseMLLLogit.class).labels(), labelsMap),
        mapTarget(VecTools.toIntSeq(predict), labelsMap)
    );
    if (oneLine) {
      return confusionMatrix.oneLineReport();
    } else {
      final StringBuilder sb = new StringBuilder();
      sb.append("\n==========").append(prefixComment).append(StringUtils.repeatWithDelimeter("", "=", 100 - 10 - prefixComment.length())).append("\n");
      sb.append(confusionMatrix.toSummaryString());
      sb.append("\n");
      sb.append(confusionMatrix.toClassDetailsString());
      sb.append(StringUtils.repeatWithDelimeter("", "=", 100));
      return sb.toString();
    }
  }

  //only for boosting of MultiClassModel models
  public static MultiClassModel joinBoostingResults(Ensemble ensemble) {
    if (ensemble.last() instanceof MultiClassModel) {
      final int internModelCount = ((MultiClassModel) ensemble.last()).getInternModel().ydim();
      final Func[] joinedModels = new Func[internModelCount];
      final Func[][] transpose = new Func[internModelCount][ensemble.size()];
      for (int iter = 0; iter < ensemble.size(); iter++) {
        final MultiClassModel model = (MultiClassModel) ensemble.models[iter];
        final Func[] sourceFunctions = model.getInternModel().dirs();
        for (int c = 0; c < internModelCount; c++) {
          transpose[c][iter] = sourceFunctions[c];
        }
      }
      for (int i = 0; i < joinedModels.length; i++) {
        joinedModels[i] = new FuncEnsemble(transpose[i], ensemble.weights);
      }
      return new MultiClassModel(joinedModels);
    }
    else
      throw new ClassCastException("Ensemble object does not contain MultiClassModel objects");
  }

  public static IntSeq mapTarget(final IntSeq intTarget, final TIntIntMap mapping) {
    final int[] mapped = new int[intTarget.length()];
    for (int i = 0; i < intTarget.length(); i++) {
      mapped[i] = mapping.get(intTarget.at(i));
    }
    return new IntSeq(mapped);
  }
}
