package com.spbsu.exp.tools;

import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.models.MultiClassModel;
import com.spbsu.ml.models.ObliviousTree;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * User: qdeee
 * Date: 25.08.14
 */
public class FeatureFrequency {
  public static int[] calcFeaturesFreq(final Trans model) {
    final int[] counts = new int[model.xdim()];
    callCorrespond(model, counts);
    return counts;
  }

  public static Pair<int[], int[]> getSortedFeaturesFreq(final Trans model) {
    final int[] counts = calcFeaturesFreq(model);
    final int[] idxs = ArrayTools.sequence(0, counts.length);
    ArrayTools.parallelSort(counts, idxs);
    return Pair.create(idxs, counts);
  }

  private static void callCorrespond(final Trans model, final int[] counts) {
    try {
      final Method method = FeatureFrequency.class.getDeclaredMethod("calcFreq", model.getClass(), counts.getClass());
      method.setAccessible(true);
      method.invoke(null, model, counts);
    }
    catch (NoSuchMethodException ignored) {}
    catch (InvocationTargetException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  private static void calcFreq(final ObliviousTree tree, final int[] counts) {
    final int depth = Math.min(tree.features().size(), 2); //scan only first two levels
    for (int i = 0; i < depth; i++) {
      counts[tree.features().get(i).findex]++;
    }
  }

  private static void calcFreq(final MultiClassModel multiClassModel, final int[] counts) {
    final Func[] dirs = multiClassModel.getInternModel().dirs();
    for (Func func : dirs) {
      callCorrespond(func, counts);
    }
  }

  private static void calcFreq(final Ensemble ensemble, final int[] counts) {
    final Trans[] models = ensemble.models;
    for (Trans model : models) {
      callCorrespond(model, counts);
    }
  }

  private static void calcFreq(final FuncEnsemble ensemble, final int[] counts) {
    calcFreq((Ensemble) ensemble, counts);
  }
}
