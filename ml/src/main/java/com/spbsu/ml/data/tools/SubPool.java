package com.spbsu.ml.data.tools;


import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.PoolFeatureMeta;

/**
 * User: solar
 * Date: 11.07.14
 * Time: 22:57
 */
public class SubPool<I extends DSItem> extends Pool<I> {

  public SubPool(final Pool<I> original, final int[] indices) {
    super(original.meta,
        ArrayTools.cut(original.items, indices),
        cutFeatures(original.features, indices),
        cutFeatures(original.targets.toArray(new Pair[original.targets.size()]), indices));
  }

  private static <T extends PoolFeatureMeta> Pair<? extends T, ? extends Seq<?>>[] cutFeatures(final Pair<? extends T, ? extends Seq<?>>[] original, final int[] indices) {
    @SuppressWarnings("unchecked")
    final Pair<T, Seq<?>>[] result = new Pair[original.length];
    for (int i = 0; i < original.length; i++) {
      result[i] = Pair.<T, Seq<?>>create(original[i].first, ArrayTools.cut(original[i].getSecond(), indices));
    }
    return result;
  }
}
