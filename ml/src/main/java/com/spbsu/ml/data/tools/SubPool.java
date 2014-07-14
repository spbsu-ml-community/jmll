package com.spbsu.ml.data.tools;


import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.FeatureMeta;

/**
 * User: solar
 * Date: 11.07.14
 * Time: 22:57
 */
public class SubPool<I extends DSItem> extends Pool<I> {
  public final int[] indices;

  public SubPool(final Pool<I> original, int[] indices) {
    super(original.meta,
        ArrayTools.cut(original.items, indices),
        cutFeatures(original.features, indices),
        ArrayTools.cut(original.target, indices));
    this.indices = indices;
  }

  private static Pair<FeatureMeta, ? extends Seq<?>>[] cutFeatures(Pair<FeatureMeta, ? extends Seq<?>>[] original, int[] indices) {
    Pair<FeatureMeta, Seq<?>>[] result = new Pair[original.length];
    for (int i = 0; i < original.length; i++) {
      result[i] = Pair.<FeatureMeta, Seq<?>>create(original[i].first, ArrayTools.cut(original[i].getSecond(), indices));
    }
    return result;
  }
}
