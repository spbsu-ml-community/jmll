package com.expleague.text;

import com.expleague.commons.func.Functions;
import com.expleague.commons.io.codec.seq.ListDictionary;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.ml.data.tools.FeatureSet;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.impl.FeatureMetaImpl;
import gnu.trove.list.array.TIntArrayList;

public class VGramBM25FeatureSet extends FeatureSet.Stub<TextItem> {
  private final ListDictionary<Character> dict;
  private final TIntArrayList freqs;
  private final int total;
  private TextItem current;

  public VGramBM25FeatureSet(ListDictionary<Character> dict, TIntArrayList freqs) {
    super(generateWordFeatures(dict));
    this.dict = dict;
    this.freqs = freqs;
    this.total = freqs.sum();
  }

  @Override
  public int dim() {
    return dict.size();
  }

  @Override
  public void accept(TextItem item) {
    this.current = item;
  }

  @Override
  public Vec advanceTo(Vec to) {
    dict.parse(current.text(), freqs, total)
        .stream()
        .forEach(idx -> to.adjust(idx, 1));

    final double docLength = VecTools.l1(to);
    final VecIterator iter = to.nonZeroes();
    while (iter.advance()) {
      final double x = iter.value();
      final double bm25 = x / (x + 2. + docLength / 300.);
      iter.setValue(bm25 * (1./Math.log(freqs.get(iter.index()) + 2.)));
    }
    return to;
  }

  private static FeatureMeta[] generateWordFeatures(ListDictionary<Character> dict) {
    return dict.alphabet().stream().map(Functions.cast(CharSeq.class)).map(word ->
        new FeatureMetaImpl("vg_" + CharSeqTools.replace(word, " ", "_"), "VGram [" + word + "]", FeatureMeta.ValueType.SPARSE_VEC)
    ).toArray(FeatureMeta[]::new);
  }
}
