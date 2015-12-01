package com.spbsu.ml.models.gpf.weblogmodel;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.models.gpf.AttractivenessModel;
import com.spbsu.ml.models.gpf.Session;
import gnu.trove.list.array.TIntArrayList;

/**
 * Created by irlab on 07.10.2014.
 */
public class SessionV1AttractivenessModel implements AttractivenessModel<BlockV1> {
  public final int NFEATS = 38;
  public final int MAX_NONZERO_FEATS = 8;

  private Func f_model;

  @Override
  public double eval_f(final Session<BlockV1> ses, final int s, final int e, final int click_s) {
    final TIntArrayList nonzeroFeats = getNonzeroFeats(ses.getBlock(s), ses.getBlock(e), click_s);
    final Vec features = new ArrayVec(NFEATS);
    for (int j = 0; j < nonzeroFeats.size(); j++)
      features.set(nonzeroFeats.getQuick(j), 1.);
    return f_model.value(features);
  }

  @Override
  public SparseVec feats(final Session<BlockV1> ses, final int s, final int e, final int click_s) {
    final TIntArrayList nonzeroFeats = getNonzeroFeats(ses.getBlock(s), ses.getBlock(e), click_s);
    final double[] ones = new double[nonzeroFeats.size()];
    for (int i = 0; i < ones.length; i++)
      ones[i] = 1.;
    final SparseVec features = new SparseVec(NFEATS, nonzeroFeats.toArray(), ones);
    return features;
  }

  /**
   * Модель аттрактивности $f$ зависит от вектора булевских фич.
   * Функция getNonzeroFeats возвращает список индексов ненулевых фич.
   * @param bs - block start
   * @param be - block end
   * @param click_s in {0,1} - флаг клика на блоке bs
   * @return список индексов ненулевых фич
   */
  private TIntArrayList getNonzeroFeats(final BlockV1 bs, final BlockV1 be, final int click_s) {
    final TIntArrayList ret = new TIntArrayList(MAX_NONZERO_FEATS);
    int index = 0;
    //  бинарные фичи для каждого типа блока $e$: [WEB, NEWS, IMAGES, DIRECT, VIDEO, OTHER]
    if (be.resultType != null)
      ret.add(index + be.resultType.ordinal());
    index += 6; // Session.ResultType.values().length
    // бинарные фичи асессорской релевантности для блока $e$, 5 градаций + NOT\_ASED
    if (be.resultGrade != null)
      ret.add(index + be.resultGrade.ordinal());
    index += 6; // Session.ResultGrade.values().length
    if (bs.blockType == Session.BlockType.RESULT && be.blockType == Session.BlockType.RESULT) {
      // бинарная фича ``переход к следующему блоку'' $R_i \to R_{i+1}$
      if (bs.position + 1 == be.position)
        ret.add(index);
      // бинарная фича ``переход к предыдущему блоку'' $R_i \to R_{i-1}$
      if (bs.position - 1 == be.position)
        ret.add(index + 1);
    }
    index += 2;
    // бинарные фичи аттрактивности позиции $R_i$ для переходов $Q \to R_i$
    if (bs.blockType == Session.BlockType.Q && be.blockType == Session.BlockType.RESULT && be.position <= 9)
      ret.add(index + be.position);
    index += 10;
    // бинарные фичи аттрактивности позиции $R_i$ для переходов $S \to R_i$
    if (bs.blockType == Session.BlockType.S && be.blockType == Session.BlockType.RESULT && be.position <= 9)
      ret.add(index + be.position);
    index += 10;
    // бинарные фичи для переходов, в зависимости от наличия клика $c\in \{0,1\}$: $(R_i, c) \to E$, $(R_i, c) \to S$
    if (bs.blockType == Session.BlockType.RESULT && be.blockType == Session.BlockType.E)
      ret.add(index + click_s);
    index += 2;
    if (bs.blockType == Session.BlockType.RESULT && be.blockType == Session.BlockType.S)
      ret.add(index + click_s);
    index += 2;

    if (index != NFEATS) throw new AssertionError("index != NFEATS, " + index + ", " + NFEATS);
    return ret;
  }

  public Func getF_model() {
    return f_model;
  }

  public void setF_model(final Func f_model) {
    this.f_model = f_model;
  }

  @Override
  public int getEdgeFeatCount() {
    return NFEATS;
  }
}
