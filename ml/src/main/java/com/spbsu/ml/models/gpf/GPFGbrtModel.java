package com.spbsu.ml.models.gpf;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.Func;
import gnu.trove.list.array.TIntArrayList;

import java.util.*;

/**
 * User: irlab
 * Date: 14.05.14
 */
public class GPFGbrtModel extends GPFModel.Stub implements GPFModel {
  public final int NFEATS = 38;
  public final int MAX_NONZERO_FEATS = 8;

  // parameters for eval_L_and_Gradient
  public double PRUNE_A_THRESHOLD = 1E-5; //0.01;

  private VecBasedMx clickProbability = new VecBasedMx(Session.ResultType.values().length, Session.ResultGrade.values().length);
  
  Func f_model = new Func.Stub() {
    @Override
    public double value(Vec x) {
      if (x.dim() != NFEATS) throw new IllegalArgumentException("x.dim() != NFEATS: " + x.dim() + " != " + NFEATS);
      return 1;  
    }

    @Override
    public int dim() {
      return NFEATS;
    }
  };
  
  public GPFGbrtModel() {
  }

  public GPFGbrtModel(GPFGbrtModel model) {
    this.MAX_PATH_LENGTH = model.MAX_PATH_LENGTH;
    this.PRUNE_A_THRESHOLD = model.PRUNE_A_THRESHOLD;
    this.clickProbability = new VecBasedMx(model.clickProbability);
  }

  public double eval_f(Session ses, int s, int e, int click_s) {
    TIntArrayList nonzeroFeats = getNonzeroFeats(ses.getBlock(s), ses.getBlock(e), click_s);
    Vec features = new ArrayVec(NFEATS);
    for (int j = 0; j < nonzeroFeats.size(); j++)
      features.set(nonzeroFeats.getQuick(j), 1.);
    return f_model.value(features);
  }

  public static class SessionFeatureRepresentation {
    final Session ses;
    final int f_count; // number of different attractiveness functions "f", roughly it is (number of edges) * (2 = click_s)
    final ArrayList<FeatureKey> keys; // f_count rows, 4 columns; keys.get(i) == [start, end, click_s]
    final VecBasedMx features;   // f_count rows, NFEATS columns; features.row(i) is a feature representation of attractiveness
    final int blocks_length; // ses.getBlocks().length;

    private final Map<FeatureKey, Integer> keys_hash; // i == keys_hash.get(keys.get(i))

    static class FeatureKey {
      int s;
      int e;
      int click_s;

      FeatureKey(int s, int e, int click_s) {
        this.s = s;
        this.e = e;
        this.click_s = click_s;
      }

      @Override
      public String toString() {
        return "FeatureKey{(" + s + "," + click_s + "->" + e + '}';
      }

      @Override
      public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        FeatureKey that = (FeatureKey) o;

        if (click_s != that.click_s) return false;
        if (e != that.e) return false;
        if (s != that.s) return false;

        return true;
      }

      @Override
      public int hashCode() {
        int result = s;
        result = 31 * result + e;
        result = 31 * result + click_s;
        return result;
      }
    }

    public SessionFeatureRepresentation(Session ses, GPFGbrtModel model) {
      this.ses = ses;
      keys = new ArrayList<FeatureKey>();

      for (int i = 0; i < ses.getBlocks().length; i++) {
        for (int j: ses.getEdgesFrom(i))
          keys.add(new FeatureKey(i, j, 0));
        if (ses.hasClickOn(i)) {
          for (int j: ses.getEdgesFrom(i))
            keys.add(new FeatureKey(i, j, 1));
        }
      }
      keys.trimToSize();
      f_count = keys.size();

      features = new VecBasedMx(f_count, model.NFEATS);
      for (int i = 0; i < keys.size(); i++) {
        FeatureKey key = keys.get(i);
        TIntArrayList nonzeroFeats = model.getNonzeroFeats(ses.getBlock(key.s), ses.getBlock(key.e), key.click_s);
        for (int j = 0; j < nonzeroFeats.size(); j++)
          features.set(i, nonzeroFeats.getQuick(j), 1.);
      }

      keys_hash = new HashMap<FeatureKey, Integer>();
      for (int i = 0; i < keys.size(); i++) {
        keys_hash.put(keys.get(i), i);
      }

      blocks_length = ses.getBlocks().length;
    }
  }

  /**
   * Модель аттрактивности $f$ зависит от вектора булевских фич.
   * Функция getNonzeroFeats возвращает список индексов ненулевых фич.
   * @param bs - block start
   * @param be - block end
   * @param click_s in {0,1} - флаг клика на блоке bs
   * @return список индексов ненулевых фич
   */
  private TIntArrayList getNonzeroFeats(Session.Block bs, Session.Block be, int click_s) {
    TIntArrayList ret = new TIntArrayList(MAX_NONZERO_FEATS);
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

  static class SessionGradientValue {
    double loglikelihood = 0.; // log(P(session)) == sum of loglikelihood over observations
    ArrayVec gradient;
    int nObservations = 0;
  }

  SessionGradientValue eval_L_and_dL_df(SessionFeatureRepresentation sesf, final boolean do_eval_gradient, Vec f) {
    SessionGradientValue ret = new SessionGradientValue();

    Session ses = sesf.ses;
    ret.nObservations = ses.getClick_indexes().length + 1;
    if (do_eval_gradient)
      ret.gradient = new ArrayVec(sesf.f_count);

    // 1 & для каждой пары блоков $i$, $j$ вычислить $f(i,j)$; третья координата - наличие клика c_i
    if (f != null) {
      if (f.dim() != sesf.f_count)
        throw new IllegalArgumentException("f.dim() != sesf.f_count:" + f.dim() + " != " + sesf.f_count);
    } else {
      f = new ArrayVec(sesf.f_count);
      for (int i = 0; i < sesf.f_count; i++)
        f.set(i, f_model.value(sesf.features.row(i)));
    }

    // 2 & для каждого блока $i$ вычислить норму $\sum_k f(i, k)$; Вторая координата - наличие клика c_i
    Mx sum_f_i_k = new VecBasedMx(sesf.blocks_length, 2);
    for (int i = 0; i < sesf.f_count; i++) {
      SessionFeatureRepresentation.FeatureKey key = sesf.keys.get(i);
      sum_f_i_k.adjust(key.s, key.click_s, f.get(i));
    }

    // 3 & для каждого блока $i$ вычислить $P(c=0|r_i)$
    Session.Block[] blocks = ses.getBlocks();
    double[] P_noclick_i = new double[sesf.blocks_length];
    for (int i = 0; i < sesf.blocks_length; i++)
      P_noclick_i[i] = 1. - getClickGivenViewProbability(blocks[i]);

    // 4 & для каждой пары блоков $i$, $j$ вычислить $P(i \to j)$; третья координата - наличие клика c_i
    ArrayVec P_i_j = new ArrayVec(sesf.f_count);
    for (int i = 0; i < sesf.f_count; i++) {
      SessionFeatureRepresentation.FeatureKey key = sesf.keys.get(i);
      P_i_j.set(i, f.get(i) / sum_f_i_k.get(key.s, key.click_s));
    }

    VecBasedMx dPji_dfjm = null; // dPji_dfjm(pi, fi) = dP(sesf.keys[pi]{ s, click_s -> e }) / df(sesf.keys[fi]{ s, click_s -> e })
    if (do_eval_gradient) {
      // 8 & для каждой пары блоков $i$, $j$ вычислить $\frac{d}{df}P(i,j)$
      dPji_dfjm = new VecBasedMx(sesf.f_count, sesf.f_count); // fPji_dfjm(pi, fi) = dP(sesf.keys[pi]{ s, click_s -> e }) / df(sesf.keys[fi]{ s, click_s -> e })
      for (int pi = 0; pi < sesf.f_count; pi++) {
        double p_pi = P_i_j.get(pi);
        double val2 = - p_pi * p_pi / f.get(pi);
        SessionFeatureRepresentation.FeatureKey pi_key = sesf.keys.get(pi);
        for (int j: ses.getEdgesFrom(pi_key.s)) {
          int fi = sesf.keys_hash.get(new SessionFeatureRepresentation.FeatureKey(pi_key.s, j, pi_key.click_s));
          dPji_dfjm.set(pi, fi, val2);
        }
        dPji_dfjm.adjust(pi, pi, p_pi / f.get(pi));
      }
    }

    // далее -- вычисления для каждого клика в отдельности
    int[] observations = new int[ses.getClick_indexes().length + 2];
    observations[0] = Session.Q_ind;
    for (int i = 0; i < ses.getClick_indexes().length; i++)
      observations[i+1] = ses.getClick_indexes()[i];
    observations[observations.length - 1] = Session.E_ind;

    for (int eindex = 1; eindex < observations.length; eindex++) {
      // 5 & для всех блоков $i$ и длин $t$ вычислить $A(s,i,t)$
      int s = observations[eindex - 1];
      int e = observations[eindex];
      int click_s = s == 0 ? 0 : 1;
      // $A(i,j,t)$ вероятность пройти без из блока $i$ в блок $j$ за $t$ шагов не сделав по пути ни одного клика
      VecBasedMx A = new VecBasedMx(blocks.length, MAX_PATH_LENGTH + 1);

      // Критерий ранней остановки вычисления матрицы $A$: если
      //   A(s,i,t+1) < \varepsilon \max_{i,u<=t} A(s,i,u)
      // то все пути длины $>t$ маловероятны по сравнению с путями длины $\le t$, и значения $A$ и $\frac{dA}{d\Theta}$ можно не вычислять
      int max_path_length_pruned = MAX_PATH_LENGTH;

      // A(s,i,1) =& P(s\to i) \cdot P(c=0 | i)              &\quad \forall i
      double max_A_lte_t = 0.;
      for (int i: ses.getEdgesFrom(s)) {
        double val = P_i_j.get(sesf.keys_hash.get(new SessionFeatureRepresentation.FeatureKey(s, i, click_s))) * P_noclick_i[i];
        A.set(i, 1, val);
        max_A_lte_t = Math.max(max_A_lte_t, val);
      }
      for (int t = 1; t < MAX_PATH_LENGTH; t++) {
        double max_A_tp1 = 0;
        for (int i = 0; i < blocks.length; i++) {
          double val = 0;
          for (int j: ses.getEdgesTo(i))
            val += A.get(j, t) * P_i_j.get(sesf.keys_hash.get(new SessionFeatureRepresentation.FeatureKey(j, i, 0)));
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
      double sumA_e_t = 0.;
      for (int t = 1; t <= max_path_length_pruned; t++)
        sumA_e_t += A.get(e, t);
      double observation_prob = sumA_e_t * (1 - P_noclick_i[e]) / P_noclick_i[e];
      ret.loglikelihood += Math.log(observation_prob);

      if (do_eval_gradient) {
        // 9 & для всех блоков $i$ и длин $t$ вычислить $\frac{dA}{df_{fi}}(s,i,t)$
        Tensor3 dA_df = new Tensor3(blocks.length, max_path_length_pruned + 1, sesf.f_count);
        // \frac{dA}{d\Theta}(s,i,1) =& P(c=0 | i) \cdot \frac{d}{d\Theta}P(s\to i) + P(s\to i) \cdot \frac{d}{d\Theta} P(c=0|i) \qquad \forall i
        for (int i: ses.getEdgesFrom(s)) {
          ArrayVec val = (ArrayVec)dPji_dfjm.row(sesf.keys_hash.get(new SessionFeatureRepresentation.FeatureKey(s, i, click_s)));
          val.scale(P_noclick_i[i]);
          dA_df.setRow(i, 1, val);
        }
        // \frac{dA}{df}(s,i,t+1) =&
        //  P(c=0|i) \sum_j \left( \frac{dA}{df}(s,j,t) \cdot P(j\to i)
        //       + A(s,j,t) \cdot \frac{d}{d\Theta} P(j\to i) \right)
        ArrayVec sum = new ArrayVec(sesf.f_count);
        ArrayVec val1 = new ArrayVec(sesf.f_count);
        ArrayVec val2 = new ArrayVec(sesf.f_count);
        for (int t = 1; t < max_path_length_pruned; t++) {
          for (int i = 0; i < blocks.length; i++) {
            sum.fill(0);
            for (int j: ses.getEdgesTo(i)) {
              int f_ji0_index = sesf.keys_hash.get(new SessionFeatureRepresentation.FeatureKey(j, i, 0));
              // sum += A.get(j, t) * P_i_j.get(j, i, 0);
              val1.assign(dA_df.getRow(j, t));
              val1.scale(P_i_j.get(f_ji0_index));

              val2.assign((ArrayVec)dPji_dfjm.row(f_ji0_index));
              val2.scale(A.get(j, t));

              sum.add(val1);
              sum.add(val2);
            }
            sum.scale(P_noclick_i[i]);
            // A.set(i, t + 1, sum * P_noclick_i[i]);
            dA_df.setRow(i, t + 1, sum);
          }
        }

        // 10 & вычислить градиент $\frac{d}{d\Theta} \log P(O_{d,\nu})$
        ArrayVec dPlogO_df = new ArrayVec(sesf.f_count);
        //        for (int t = 1; t <= MAX_PATH_LENGTH; t++)
        //          observation_prob += A.get(e, t);
        //        observation_prob *= (1 - P_noclick_i[e]) / P_noclick_i[e];
        //        ret.observation_probabilities.set(e-1, observation_prob);
        for (int t = 1; t <= max_path_length_pruned; t++)
          dPlogO_df.add(dA_df.getRow(e, t));
        dPlogO_df.scale( 1./sumA_e_t );
        ret.gradient.add(dPlogO_df);
      }
    }

    return ret;
  }

  public void trainClickProbability(List<Session> dataset) {
    VecBasedMx shows = new VecBasedMx(Session.ResultType.values().length, Session.ResultGrade.values().length);
    VecBasedMx clicks = new VecBasedMx(Session.ResultType.values().length, Session.ResultGrade.values().length);
    for (Session ses: dataset) {
      Session.Block block1 = ses.getBlock(Session.R0_ind);
      shows.adjust(block1.resultType.ordinal(), block1.resultGrade.ordinal(), 1);
      if (ses.hasClickOn(Session.R0_ind))
        clicks.adjust(block1.resultType.ordinal(), block1.resultGrade.ordinal(), 1);
    }

    double[] shows_result_type = new double[Session.ResultType.values().length];
    double[] clicks_result_type = new double[Session.ResultType.values().length];
    double shows_all = 0;
    double clicks_all = 0;
    for (int i = 0; i < Session.ResultType.values().length; i++) {
      for (int j = 0; j < Session.ResultGrade.values().length; j++) {
        shows_result_type[i] += shows.get(i, j);
        clicks_result_type[i] += clicks.get(i, j);
      }
      shows_all += shows_result_type[i];
      clicks_all += clicks_result_type[i];
    }

    double ctr_all = clicks_all / shows_all;
    for (int i = 0; i < Session.ResultType.values().length; i++) {
      double prob_click_result_type = (clicks_result_type[i] + 10 * ctr_all) / (shows_result_type[i] + 10);
      for (int j = 0; j < Session.ResultGrade.values().length; j++) {
        double prob = (clicks.get(i, j) + 10 * prob_click_result_type) / (shows.get(i, j) + 10);
        clickProbability.set(i, j, prob);
      }
    }
  }

  public double getClickGivenViewProbability(Session.Block b) {
    switch (b.blockType) {
      case RESULT:
        return clickProbability.get(b.resultType.ordinal(), b.resultGrade.ordinal());
      case Q:
        return 1. - 1e-6; // always observed
      case S:
        return 0; // never observed
      case E:
        return 1. - 1e-6; // always observed
    }
    throw new IllegalStateException("unknown ResultType: " + b);
  }

  public String explainTheta() {
    return "NOT IMPLEMENTED";
  }

  @Override
  public String toString() {
    return "NOT IMPLEMENTED";
  }
}
