package com.spbsu.ml.models.gpf;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.ml.models.gpf.weblogmodel.BlockV1;
import gnu.trove.list.array.TIntArrayList;
import org.apache.commons.lang3.NotImplementedException;


import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;

/**
 * User: irlab
 * Date: 14.05.14
 */
public class GPFLinearModel extends GPFModel.Stub<BlockV1> implements GPFModel<BlockV1>, ClickProbabilityModel<BlockV1> {
  public final int NFEATS = 38;
  public final int MAX_NONZERO_FEATS = 8;

  // parameters for eval_L_and_Gradient
  public double PRUNE_A_THRESHOLD = 0.01;

  public ArrayVec theta = new ArrayVec(NFEATS);

  private VecBasedMx clickProbability = new VecBasedMx(BlockV1.ResultType.values().length, BlockV1.ResultGrade.values().length);

  public GPFLinearModel() {
  }

  public GPFLinearModel(final GPFLinearModel model) {
    this.MAX_PATH_LENGTH = model.MAX_PATH_LENGTH;
    this.PRUNE_A_THRESHOLD = model.PRUNE_A_THRESHOLD;
    this.theta = new ArrayVec(model.theta.toArray());
    this.clickProbability = new VecBasedMx(model.clickProbability);
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

  @Override
  public double eval_f(final Session<BlockV1> ses, final int s, final int e, final int click_s) {
    return eval_f(getNonzeroFeats(ses.getBlock(s), ses.getBlock(e), click_s));
  }

  protected double eval_f(final TIntArrayList nonzeroFeats) {
    // Функция аттрактивности перехода s->e f = <F(s,e),w> - линейная функция от фич пары блоков и предшествовавшего клика
    double ret = 0.;
    for (int i = 0; i < nonzeroFeats.size(); i++)
      ret += theta.get(nonzeroFeats.getQuick(i));
    ret = Math.exp(ret);
    return ret;
  }

  protected ArrayVec eval_df_dTheta(final BlockV1 bs, final BlockV1 be, final int click_s) {
    final TIntArrayList nonzeroFeats = getNonzeroFeats(bs, be, click_s);
    final double f_value = eval_f(nonzeroFeats);
    return eval_df_dTheta(nonzeroFeats, f_value);
  }

  protected ArrayVec eval_df_dTheta(final TIntArrayList nonzeroFeats, final double f_value) {
    // Функция аттрактивности перехода s->e f = <F(s,e),w> - линейная функция от фич пары блоков и предшествовавшего клика
    final ArrayVec ret = new ArrayVec(NFEATS);
    for (int i = 0; i < nonzeroFeats.size(); i++)
      ret.set(nonzeroFeats.getQuick(i), f_value);
    return ret;
  }

  class GradientValue {
    Vec observation_probabilities; // size is (ses.clicks.length + 1)

    // градиент loglikelihood $\frac{d}{d\Theta} \log P(O_{d,\nu})$
    VecBasedMx gradient; // (ses.clicks.length + 1) \times NFEATS
  }

  GradientValue eval_L_and_Gradient(final Session<BlockV1> ses, final boolean do_eval_gradient) {
    final GradientValue ret = new GradientValue();
    ret.observation_probabilities = new ArrayVec(ses.getClick_indexes().length + 1);
    if (do_eval_gradient)
      ret.gradient = new VecBasedMx(ses.getClick_indexes().length + 1, NFEATS);

    final BlockV1[] blocks = ses.getBlocks();

    // 1 & для каждой пары блоков $i$, $j$ вычислить $f(i,j)$; третья координата - наличие клика c_i
    final Tensor3 f = new Tensor3(blocks.length, blocks.length, 2);
    for (int i = 0; i < blocks.length; i++) {
      for (final int j: ses.getEdgesFrom(i))
        f.set(i, j, 0, eval_f(ses, i, j, 0));
      if (ses.hasClickOn(i)) {
        for (final int j: ses.getEdgesFrom(i))
          f.set(i, j, 1, eval_f(ses, i, j, 1));
      }
    }
    f.set(Session.E_INDEX, Session.E_INDEX, 0, 1.);
    f.set(Session.E_INDEX, Session.E_INDEX, 1, 1.);

    // 2 & для каждого блока $i$ вычислить норму $\sum_k f(i, k)$; Вторая координата - наличие клика c_i
    final Mx sum_f_i_k = new VecBasedMx(blocks.length, 2);
    for (int i = 0; i < blocks.length; i++) {
      double sum = 0;
      for (final int k: ses.getEdgesFrom(i))
        sum += f.get(i, k, 0);
      sum_f_i_k.set(i, 0, sum);

      if (ses.hasClickOn(i) || i == Session.E_INDEX) {
        sum = 0;
        for (final int k: ses.getEdgesFrom(i))
          sum += f.get(i, k, 1);
        sum_f_i_k.set(i, 1, sum);
      }
    }

    // 3 & для каждого блока $i$ вычислить $P(c=0|r_i)$
    final double[] P_noclick_i = new double[blocks.length];
    for (int i = 0; i < blocks.length; i++) {
      switch (blocks[i].blockType) {
        case RESULT:
          P_noclick_i[i] = 1. - getClickGivenViewProbability(blocks[i]);
          break;
        case Q:
          P_noclick_i[i] = 1e-6; // always observed
          break;
        case S:
          P_noclick_i[i] = 1; // never observed
          break;
        case E:
          P_noclick_i[i] = 1e-6; // always observed
          break;
      }
    }

    // 4 & для каждой пары блоков $i$, $j$ вычислить $P(i \to j)$; третья координата - наличие клика c_i
    final Tensor3 P_i_j = new Tensor3(blocks.length, blocks.length, 2);
    for (int i = 0; i < blocks.length; i++) {
      for (final int j: ses.getEdgesFrom(i))
        P_i_j.set(i, j, 0, f.get(i, j, 0) / sum_f_i_k.get(i, 0));
      if (ses.hasClickOn(i)) {
        for (final int j: ses.getEdgesFrom(i))
          P_i_j.set(i, j, 1, f.get(i, j, 1) / sum_f_i_k.get(i, 1));
      }
    }

    Tensor4 df_dTheta = null;
    Tensor3 sum_df_dTheta_i_k = null;
    Tensor4 dPij_dTheta = null;
    if (do_eval_gradient) {
      // 7 & для каждой пары блоков $i$, $j$ вычислить $\frac{df}{d\Theta}(i,j)$
      df_dTheta = new Tensor4(blocks.length, blocks.length, 2, NFEATS);
      for (int i = 0; i < blocks.length; i++) {
        for (final int j: ses.getEdgesFrom(i))
          df_dTheta.setRow(i, j, 0, eval_df_dTheta(blocks[i], blocks[j], 0));
        if (ses.hasClickOn(i)) {
          for (final int j: ses.getEdgesFrom(i))
            df_dTheta.setRow(i, j, 1, eval_df_dTheta(blocks[i], blocks[j], 1));
        }
      }
      // df_dTheta(E -> E) already set to 0
      //df_dTheta.setRow(Session.E_INDEX, Session.E_INDEX, 0, new ArrayVec(model.NFEATS));
      //df_dTheta.setRow(Session.E_INDEX, Session.E_INDEX, 1, new ArrayVec(model.NFEATS));

      // для каждого блока $i$ вычисляем сумму $\sum_k df_dTheta(i,k)$
      sum_df_dTheta_i_k = new Tensor3(blocks.length, 2, NFEATS);
      for (int i = 0; i < blocks.length; i++) {
        ArrayVec sum = new ArrayVec(NFEATS);
        for (final int k: ses.getEdgesFrom(i))
          sum.add(df_dTheta.getRow(i, k, 0));
        sum_df_dTheta_i_k.setRow(i, 0, sum);

        if (ses.hasClickOn(i) || i == Session.E_INDEX) {
          sum = new ArrayVec(NFEATS);
          for (final int k: ses.getEdgesFrom(i))
            sum.add(df_dTheta.getRow(i, k, 1));
          sum_df_dTheta_i_k.setRow(i, 1, sum);
        }
      }

      // 8 & для каждой пары блоков $i$, $j$ вычислить $\frac{d}{d\Theta}P(i,j)$
      dPij_dTheta = new Tensor4(blocks.length, blocks.length, 2, NFEATS);
      final ArrayVec val1 = new ArrayVec(NFEATS);
      for (int i = 0; i < blocks.length; i++) {
        for (final int j: ses.getEdgesFrom(i)) {
          val1.assign(sum_df_dTheta_i_k.getRow(i, 0));
          val1.scale( - f.get(i, j, 0) / sum_f_i_k.get(i, 0) );
          val1.add( df_dTheta.getRow(i, j, 0) );
          val1.scale(1. / sum_f_i_k.get(i, 0));
          dPij_dTheta.setRow(i, j, 0, val1);
        }

        if (ses.hasClickOn(i)) {
          for (final int j: ses.getEdgesFrom(i)) {
            val1.assign(sum_df_dTheta_i_k.getRow(i, 1));
            val1.scale( - f.get(i, j, 1) / sum_f_i_k.get(i, 1) );
            val1.add( df_dTheta.getRow(i, j, 1) );
            val1.scale(1. / sum_f_i_k.get(i, 1));
            dPij_dTheta.setRow(i, j, 1, val1);
          }
        }
      }
    }

    // далее -- вычисления для каждого клика в отдельности
    final int[] observations = new int[ses.getClick_indexes().length + 2];
    observations[0] = Session.Q_INDEX;
    for (int i = 0; i < ses.getClick_indexes().length; i++)
      observations[i+1] = ses.getClick_indexes()[i];
    observations[observations.length - 1] = Session.E_INDEX;

    for (int eindex = 1; eindex < observations.length; eindex++) {
      // 5 & для всех блоков $i$ и длин $t$ вычислить $A(s,i,t)$
      final int s = observations[eindex - 1];
      final int e = observations[eindex];
      final int click_s = s == 0 ? 0 : 1;
      // $A(i,j,t)$ вероятность пройти без из блока $i$ в блок $j$ за $t$ шагов не сделав по пути ни одного клика
      final Mx A = new VecBasedMx(blocks.length, MAX_PATH_LENGTH + 1);

      // Критерий ранней остановки вычисления матрицы $A$: если
      //   A(s,i,t+1) < \varepsilon \max_{i,u<=t} A(s,i,u)
      // то все пути длины $>t$ маловероятны по сравнению с путями длины $\le t$, и значения $A$ и $\frac{dA}{d\Theta}$ можно не вычислять
      int max_path_length_pruned = MAX_PATH_LENGTH;

      // A(s,i,1) =& P(s\to i) \cdot P(c=0 | i)              &\quad \forall i
      double max_A_lte_t = 0.;
      for (final int i: ses.getEdgesFrom(s)) {
        final double val = P_i_j.get(s, i, click_s) * P_noclick_i[i];
        A.set(i, 1, val);
        max_A_lte_t = Math.max(max_A_lte_t, val);
      }
      for (int t = 1; t < MAX_PATH_LENGTH; t++) {
        double max_A_tp1 = 0;
        for (int i = 0; i < blocks.length; i++) {
          double val = 0;
          for (final int j: ses.getEdgesTo(i))
            val += A.get(j, t) * P_i_j.get(j, i, 0);
          val *= P_noclick_i[i];
          A.set(i, t + 1, val);
          max_A_tp1 = Math.max(max_A_tp1, val);
        }

        if (max_A_lte_t * PRUNE_A_THRESHOLD > max_A_tp1) {
          max_path_length_pruned = t;
          break;
        }
      }

      // 6 & вычислить $P(O_{d,\nu})$
      double observation_prob = 0;
      double sumA_e_t = 0.;
      for (int t = 1; t <= max_path_length_pruned; t++)
        sumA_e_t += A.get(e, t);
      observation_prob = sumA_e_t * (1 - P_noclick_i[e]) / P_noclick_i[e];
      ret.observation_probabilities.set(eindex-1, observation_prob);

      if (do_eval_gradient) {
        // 9 & для всех блоков $i$ и длин $t$ вычислить $\frac{dA}{d\Theta}(s,i,t)$
        final Tensor3 dA_dTheta = new Tensor3(blocks.length, max_path_length_pruned + 1, NFEATS);
        // \frac{dA}{d\Theta}(s,i,1) =& P(c=0 | i) \cdot \frac{d}{d\Theta}P(s\to i) + P(s\to i) \cdot \frac{d}{d\Theta} P(c=0|i) \qquad \forall i
        for (final int i: ses.getEdgesFrom(s)) {
          final ArrayVec val = dPij_dTheta.getRow(s, i, click_s);
          val.scale(P_noclick_i[i]);
          dA_dTheta.setRow(i, 1, val);
        }
        // \frac{dA}{d\Theta}(s,i,t+1) =&
        //  P(c=0|i) \sum_j \left( \frac{dA}{d\Theta}(s,j,t) \cdot P(j\to i)
        //       + A(s,j,t) \cdot \frac{d}{d\Theta} P(j\to i) \right)
        final ArrayVec sum = new ArrayVec(NFEATS);
        final ArrayVec val1 = new ArrayVec(NFEATS);
        final ArrayVec val2 = new ArrayVec(NFEATS);
        for (int t = 1; t < max_path_length_pruned; t++) {
          for (int i = 0; i < blocks.length; i++) {
            sum.fill(0);
            for (final int j: ses.getEdgesTo(i)) {
              // sum += A.get(j, t) * P_i_j.get(j, i, 0);
              val1.assign(dA_dTheta.getRow(j, t));
              val1.scale(P_i_j.get(j, i, 0));

              val2.assign(dPij_dTheta.getRow(j, i, 0));
              val2.scale(A.get(j, t));

              sum.add(val1);
              sum.add(val2);
            }
            sum.scale(P_noclick_i[i]);
            // A.set(i, t + 1, sum * P_noclick_i[i]);
            dA_dTheta.setRow(i, t + 1, sum);
          }
        }

        // 10 & вычислить градиент $\frac{d}{d\Theta} \log P(O_{d,\nu})$
        final ArrayVec dPlogO_dTheta = new ArrayVec(NFEATS);
        //        for (int t = 1; t <= MAX_PATH_LENGTH; t++)
        //          observation_prob += A.get(e, t);
        //        observation_prob *= (1 - P_noclick_i[e]) / P_noclick_i[e];
        //        ret.observation_probabilities.set(e-1, observation_prob);
        for (int t = 1; t <= max_path_length_pruned; t++)
          dPlogO_dTheta.add(dA_dTheta.getRow(e, t));
        dPlogO_dTheta.scale( 1./sumA_e_t );
        for (int l = 0; l < NFEATS; l++)
          ret.gradient.set(eindex - 1, l, dPlogO_dTheta.get(l));
      }
    }

    return ret;
  }

  @Override
  public String explainTheta() {
    final StringBuffer ret = new StringBuffer();
    ret.append("theta: {\n");

    int index = 0;
    //  бинарные фичи для каждого типа блока $e$: [WEB, NEWS, IMAGES, DIRECT, VIDEO, OTHER]
    for (final BlockV1.ResultType x: BlockV1.ResultType.values()) {
      ret.append("  w(" + x.name() + ")\t" + theta.get(index) + "\n");
      index++;
    }
    // бинарные фичи асессорской релевантности для блока $e$, 5 градаций + NOT\_ASED
    for (final BlockV1.ResultGrade x: BlockV1.ResultGrade.values()) {
      ret.append("  w(" + x.name() + ")\t" + theta.get(index) + "\n");
      index++;
    }
    // бинарная фича ``переход к следующему блоку'' $R_i \to R_{i+1}$
    ret.append("  w(R_i -> R_{i+1})\t" + theta.get(index) + "\n");
    index++;
    // бинарная фича ``переход к предыдущему блоку'' $R_i \to R_{i-1}$
    ret.append("  w(R_i -> R_{i-1})\t" + theta.get(index) + "\n");
    index++;
    // бинарные фичи аттрактивности позиции $R_i$ для переходов $Q \to R_i$
    for (int i = 0; i < 10; i++) {
      ret.append("  w(Q -> R_" + i + ")\t" + theta.get(index) + "\n");
      index++;
    }
    // бинарные фичи аттрактивности позиции $R_i$ для переходов $S \to R_i$
    for (int i = 0; i < 10; i++) {
      ret.append("  w(S -> R_" + i + ")\t" + theta.get(index) + "\n");
      index++;
    }
    // бинарные фичи для переходов, в зависимости от наличия клика $c\in \{0,1\}$: $(R_i, c) \to E$, $(R_i, c) \to S$
    ret.append("  w(noclick,R_i -> E)\t" + theta.get(index) + "\n");
    index++;
    ret.append("  w(  click,R_i -> E)\t" + theta.get(index) + "\n");
    index++;
    ret.append("  w(noclick,R_i -> S)\t" + theta.get(index) + "\n");
    index++;
    ret.append("  w(  click,R_i -> S)\t" + theta.get(index) + "\n");
    index++;
    if (index != NFEATS) throw new AssertionError("index != NFEATS, " + index + ", " + NFEATS);
    return ret.toString();
  }

  @Override
  public void trainClickProbability(final List<Session<BlockV1>> dataset) {
    final VecBasedMx shows = new VecBasedMx(BlockV1.ResultType.values().length, BlockV1.ResultGrade.values().length);
    final VecBasedMx clicks = new VecBasedMx(BlockV1.ResultType.values().length, BlockV1.ResultGrade.values().length);
    for (final Session<BlockV1> ses: dataset) {
      final BlockV1 block1 = ses.getBlock(Session.R0_INDEX);
      shows.adjust(block1.resultType.ordinal(), block1.resultGrade.ordinal(), 1);
      if (ses.hasClickOn(Session.R0_INDEX))
        clicks.adjust(block1.resultType.ordinal(), block1.resultGrade.ordinal(), 1);
    }

    final double[] shows_result_type = new double[BlockV1.ResultType.values().length];
    final double[] clicks_result_type = new double[BlockV1.ResultType.values().length];
    double shows_all = 0;
    double clicks_all = 0;
    for (int i = 0; i < BlockV1.ResultType.values().length; i++) {
      for (int j = 0; j < BlockV1.ResultGrade.values().length; j++) {
        shows_result_type[i] += shows.get(i, j);
        clicks_result_type[i] += clicks.get(i, j);
      }
      shows_all += shows_result_type[i];
      clicks_all += clicks_result_type[i];
    }

    final double ctr_all = clicks_all / shows_all;
    for (int i = 0; i < BlockV1.ResultType.values().length; i++) {
      final double prob_click_result_type = (clicks_result_type[i] + 10 * ctr_all) / (shows_result_type[i] + 10);
      for (int j = 0; j < BlockV1.ResultGrade.values().length; j++) {
        final double prob = (clicks.get(i, j) + 10 * prob_click_result_type) / (shows.get(i, j) + 10);
        clickProbability.set(i, j, prob);
      }
    }
  }

  @Override
  public double getClickGivenViewProbability(final BlockV1 b) {
    if (b.blockType == Session.BlockType.RESULT) {
      return clickProbability.get(b.resultType.ordinal(), b.resultGrade.ordinal());
    } else {
      return 0;
    }
  }

  @Override
  public String toString() {
    return "GPFLinearModel{" +
            "NFEATS=" + NFEATS +
            ", MAX_PATH_LENGTH=" + MAX_PATH_LENGTH +
            ", PRUNE_A_THRESHOLD=" + PRUNE_A_THRESHOLD +
            ", theta=" + theta +
            ", " + explainTheta() +
            '}';
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

  @Override
  public int getEdgeFeatCount() {
    return NFEATS;
  }


  @Override
  public void save(final OutputStream os) throws IOException {
    throw new NotImplementedException("not implemented");
  }

  @Override
  public void load(final InputStream is) throws IOException {
    throw new NotImplementedException("not implemented");
  }
}
