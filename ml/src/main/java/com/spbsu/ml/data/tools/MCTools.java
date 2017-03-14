package com.spbsu.ml.data.tools;

import com.spbsu.commons.func.Computable;
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
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.multiclass.MCMicroF1Score;
import com.spbsu.ml.loss.multiclass.util.ConfusionMatrix;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.models.multiclass.JoinedBinClassModel;
import com.spbsu.ml.models.multiclass.MCModel;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static java.lang.Math.max;

/**
 * User: qdeee
 * Date: 04.06.14
 */
public class MCTools {
  public static int countClasses(final IntSeq target) {
    int classesCount = 0;
    for (int i = 0; i < target.length(); i++) {
      classesCount = max(target.at(i) + 1, classesCount);
    }
    return classesCount;
  }

  /**
   * calculate classes entries counts
   * @param target normalized(!) target with class labels from {0,...,K-1}
   * @return array with counts
   */
  public static int[] classEntriesCounts(final IntSeq target) {
    final int[] counts = new int[countClasses(target)];
    for (int i = 0; i < target.length(); i++) {
      counts[target.arr[i]]++;
    }
    return counts;
  }

  public static int classEntriesCount(final IntSeq target, final int classNo) {
    int result = 0;
    for (int i = 0; i < target.length(); i++) {
      if (target.at(i) == classNo)
        result++;
    }
    return result;
  }

  public static Vec extractClassForBinary(final IntSeq target, final int classNo) {
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
  public static int[] getClassesLabels(final IntSeq target) {
    final TIntList labels = new TIntArrayList();
    for (int i = 0; i < target.length(); i++) {
      final int label = target.at(i);
      if (!labels.contains(label)) {
        labels.add(label);
      }
    }
    return labels.toArray();
  }

  public static int[] getClassLabels(final Vec target) {
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

  public static TIntObjectMap<TIntList> splitClassesIdxs(final IntSeq target) {
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

  private static double normalizeRelevance(final double y) {
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


  public static IntSeq transformRegressionToMC(final Vec regressionTarget, final int classCount, final TDoubleList borders) throws IOException {
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

  public static Pair<VecDataSet, IntSeq> loadRegressionAsMC(final String file, final int classCount, final TDoubleList borders)  throws IOException{
    final Pool<QURLItem> pool = DataTools.loadFromFeaturesTxt(file);
    return Pair.create(pool.vecData(), transformRegressionToMC(pool.target(L2.class).target, classCount, borders));
  }

  public static Mx createSimilarityMatrixParallels(final VecDataSet learn, final IntSeq target, final Metric<Vec> metric) {
    final ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    final TIntObjectMap<TIntList> indexes = splitClassesIdxs(target);
    final int k = indexes.keys().length;
    final Mx S = new VecBasedMx(k, k);
    for (int i = 0; i < k; i++) {
      final TIntList classIdxsI = indexes.get(i);

      for (int j = i; j < k; j++) {
        final TIntList classIdxsJ = indexes.get(j);
        final int iCopy = i;
        final int jCopy = j;

        executor.submit(new Runnable() {
          @Override
          public void run() {
            double value = 0.;

            for (final TIntIterator iterI = classIdxsI.iterator(); iterI.hasNext(); ) {
              final int i1 = iterI.next();
              for (final TIntIterator iterJ = classIdxsJ.iterator(); iterJ.hasNext(); ) {
                final int i2 = iterJ.next();
                value += 1 - metric.distance(learn.data().row(i1), learn.data().row(i2));
              }
            }
            value /= classIdxsI.size() * classIdxsJ.size();
            S.set(iCopy, jCopy, value);
            S.set(jCopy, iCopy, value);
          }
        });
      }
    }

    executor.shutdown();
    try {
      executor.awaitTermination(1000, TimeUnit.HOURS);
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
    return S;
  }

  public static Mx createSimilarityMatrix(final VecDataSet learn, final IntSeq target) {
    final TIntObjectMap<TIntList> indexes = splitClassesIdxs(target);
    final Metric<Vec> metric = new CosineDVectorMetric();
    final int k = indexes.keys().length;
    final Mx S = new VecBasedMx(k, k);
    for (int i = 0; i < k; i++) {
      final TIntList classIdxsI = indexes.get(i);
      for (int j = i; j < k; j++) {
        final TIntList classIdxsJ = indexes.get(j);
        double value = 0.;
        for (final TIntIterator iterI = classIdxsI.iterator(); iterI.hasNext(); ) {
          final int i1 = iterI.next();
          for (final TIntIterator iterJ = classIdxsJ.iterator(); iterJ.hasNext(); ) {
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
      return prefixComment + confusionMatrix.debug();
    } else {
      return "\n==========" + prefixComment +
          StringUtils.repeatWithDelimeter("", "=", 100 - 10 - prefixComment.length()) + "\n" +
          confusionMatrix.toSummaryString() + "\n" +
          confusionMatrix.toClassDetailsString() +
          StringUtils.repeatWithDelimeter("", "=", 100);
    }
  }

  //only for FuncJoin models
  public static FuncJoin joinBoostingResult(final Ensemble ensemble) {
    if (ensemble.last() instanceof FuncJoin) {
      final int modelsCount = ensemble.ydim();
      final Func[] joinedModels = new Func[modelsCount];
      final Func[][] transpose = new Func[modelsCount][ensemble.size()];
      for (int iter = 0; iter < ensemble.size(); iter++) {
        final FuncJoin model = (FuncJoin) ensemble.models[iter];
        final Func[] sourceFunctions = model.dirs();
        for (int c = 0; c < modelsCount; c++) {
          transpose[c][iter] = sourceFunctions[c];
        }
      }
      for (int i = 0; i < joinedModels.length; i++) {
        joinedModels[i] = new FuncEnsemble(transpose[i], ensemble.weights);
      }
      return new FuncJoin(joinedModels);
    }
    else
      throw new ClassCastException("Ensemble object does not contain FuncJoin objects");
  }

  public static IntSeq mapTarget(final IntSeq intTarget, final TIntIntMap mapping) {
    final int[] mapped = new int[intTarget.length()];
    for (int i = 0; i < intTarget.length(); i++) {
      mapped[i] = mapping.get(intTarget.at(i));
    }
    return new IntSeq(mapped);
  }

  public static void makeOneVsRestReport(final Pool<?> learn, final Pool<?> test, final JoinedBinClassModel joinedBinClassModel, final int period) {
    if (!(joinedBinClassModel.getInternModel().dirs[0] instanceof FuncEnsemble)) {
      throw new IllegalArgumentException("Provided model must contain array of FuncEnsemble objects");
    }

    final IntSeq learnLabels = learn.target(MCMicroF1Score.class).labels();
    final IntSeq testLabels = test.target(MCMicroF1Score.class).labels();

    final FuncEnsemble<?>[] perClassModels = ArrayTools.map(joinedBinClassModel.getInternModel().dirs, FuncEnsemble.class, new Computable<Trans, FuncEnsemble>() {
      @Override
      public FuncEnsemble compute(final Trans argument) {
        return (FuncEnsemble<?>) argument;
      }
    });
    final int ensembleSize = perClassModels[0].size();
    final int classesCount = perClassModels.length;

    final Mx learnCache = new VecBasedMx(learn.size(), classesCount);
    final Mx testCache = new VecBasedMx(test.size(), classesCount);

    for (int t = 0; t < ensembleSize; t += period) {
      final Func[] functions = new Func[classesCount];
      for (int c = 0; c < classesCount; c++) {
        functions[c] = new FuncEnsemble<>(
            Arrays.copyOfRange(perClassModels[c].models, t, Math.min(t + period, ensembleSize), Func[].class),
            perClassModels[c].weights.sub(0, t)
        );
      }
      final FuncJoin deltaFuncJoin = new FuncJoin(functions);

      VecTools.append(learnCache, deltaFuncJoin.transAll(learn.vecData().data()));
      VecTools.append(testCache, deltaFuncJoin.transAll(test.vecData().data()));

      final IntSeq learnPredict = convertTransOneVsRestResults(learnCache);
      final IntSeq testPredict = convertTransOneVsRestResults(testCache);

      final ConfusionMatrix learnConfusionMatrix = new ConfusionMatrix(learnLabels, learnPredict);
      final ConfusionMatrix testConfusionMatrix = new ConfusionMatrix(testLabels, testPredict);
      System.out.print("\t" + learnConfusionMatrix.oneLineReport());
      System.out.print("\t" + testConfusionMatrix.oneLineReport());
      System.out.println();
    }
  }

  private static IntSeq convertTransOneVsRestResults(final Mx trans) {
    final int[] result = new int[trans.rows()];
    for (int i = 0; i < trans.rows(); i++) {
      final Vec row = trans.row(i);
      result[i] = VecTools.argmax(row);
    }
    return new IntSeq(result);
  }
}
