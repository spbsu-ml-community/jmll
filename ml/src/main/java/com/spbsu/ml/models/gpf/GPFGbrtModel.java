package com.spbsu.ml.models.gpf;

import org.jetbrains.annotations.NotNull;


import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;


import java.util.*;

/**
 * User: irlab
 * Date: 14.05.14
 */
public class GPFGbrtModel<Blk extends Session.Block> extends GPFModel.Stub<Blk> implements GPFModel<Blk> {
  // parameters for eval_L_and_Gradient
  public double PRUNE_A_THRESHOLD = 1E-5; //0.01;

  private ClickProbabilityModel<Blk> clickProbabilityModel;
  private AttractivenessModel<Blk> attractivenessModel;
  
  public GPFGbrtModel() {
  }

  public GPFGbrtModel(final GPFGbrtModel<Blk> model) {
    this.MAX_PATH_LENGTH = model.MAX_PATH_LENGTH;
    this.PRUNE_A_THRESHOLD = model.PRUNE_A_THRESHOLD;
    this.clickProbabilityModel = model.clickProbabilityModel;
    this.attractivenessModel = model.attractivenessModel;
  }

  public static class SessionFeatureRepresentation<Blk extends Session.Block> {
    final Session ses;
    public final int f_count; // number of different attractiveness functions "f", roughly it is (number of edges) * (2 = click_s)
    final ArrayList<FeatureKey> keys; // f_count rows, 4 columns; keys.get(i) == [start, end, click_s]
    public final VecBasedMx features;   // f_count rows, NFEATS columns; features.row(i) is a feature representation of attractiveness
    final int blocks_length; // ses.getBlocks().length;

    private final Map<FeatureKey, Integer> keys_hash; // i == keys_hash.get(keys.get(i))

    static class FeatureKey {
      int s;
      int e;
      int click_s;

      FeatureKey(final int s, final int e, final int click_s) {
        this.s = s;
        this.e = e;
        this.click_s = click_s;
      }

      @Override
      public String toString() {
        return "FeatureKey{(" + s + "," + click_s + "->" + e + '}';
      }

      @Override
      public boolean equals(final Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        final FeatureKey that = (FeatureKey) o;

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

    public SessionFeatureRepresentation(final Session<Blk> ses, final AttractivenessModel<Blk> fmodel) {
      this.ses = ses;
      keys = new ArrayList<FeatureKey>();

      for (int i = 0; i < ses.getBlocks().length; i++) {
        for (final int j: ses.getEdgesFrom(i))
          keys.add(new FeatureKey(i, j, 0));
        if (ses.hasClickOn(i)) {
          for (final int j: ses.getEdgesFrom(i))
            keys.add(new FeatureKey(i, j, 1));
        }
      }
      keys.trimToSize();
      f_count = keys.size();

      features = new VecBasedMx(f_count, fmodel.getEdgeFeatCount());
      for (int i = 0; i < keys.size(); i++) {
        final FeatureKey key = keys.get(i);
        final SparseVec edgeFeatures = fmodel.feats(ses, key.s, key.e, key.click_s);
        for (final VecIterator it = edgeFeatures.nonZeroes(); it.advance(); )
          features.set(i, it.index(), it.value());
      }

      keys_hash = new HashMap<FeatureKey, Integer>();
      for (int i = 0; i < keys.size(); i++) {
        keys_hash.put(keys.get(i), i);
      }

      blocks_length = ses.getBlocks().length;
    }
  }

  static class SessionGradientValue {
    double loglikelihood = 0.; // log(P(session)) == sum of loglikelihood over observations
    ArrayVec gradient;
    int nObservations = 0;
  }

  SessionGradientValue eval_L_and_dL_df(final SessionFeatureRepresentation<Blk> sesf, final boolean do_eval_gradient, @NotNull final Vec f) {
    final SessionGradientValue ret = new SessionGradientValue();

    final Session ses = sesf.ses;
    ret.nObservations = ses.getClick_indexes().length + 1;
    if (do_eval_gradient)
      ret.gradient = new ArrayVec(sesf.f_count);

    // 1 & для каждой пары блоков $i$, $j$ вычислить $f(i,j)$; третья координата - наличие клика c_i
    if (f.dim() != sesf.f_count)
      throw new IllegalArgumentException("f.dim() != sesf.f_count:" + f.dim() + " != " + sesf.f_count);
    //  f = new ArrayVec(sesf.f_count);
    //  for (int i = 0; i < sesf.f_count; i++)
    //    f.set(i, f_model.value(sesf.features.row(i)));

    // 2 & для каждого блока $i$ вычислить норму $\sum_k f(i, k)$; Вторая координата - наличие клика c_i
    final Mx sum_f_i_k = new VecBasedMx(sesf.blocks_length, 2);
    for (int i = 0; i < sesf.f_count; i++) {
      final SessionFeatureRepresentation.FeatureKey key = sesf.keys.get(i);
      sum_f_i_k.adjust(key.s, key.click_s, f.get(i));
    }

    // 3 & для каждого блока $i$ вычислить $P(c=0|r_i)$
    final Session.Block[] blocks = ses.getBlocks();
    final double[] P_noclick_i = new double[sesf.blocks_length];
    for (int i = 0; i < sesf.blocks_length; i++)
      P_noclick_i[i] = 1. - getClickGivenViewProbability((Blk)blocks[i]);

    // 4 & для каждой пары блоков $i$, $j$ вычислить $P(i \to j)$; третья координата - наличие клика c_i
    final ArrayVec P_i_j = new ArrayVec(sesf.f_count);
    for (int i = 0; i < sesf.f_count; i++) {
      final SessionFeatureRepresentation.FeatureKey key = sesf.keys.get(i);
      P_i_j.set(i, f.get(i) / sum_f_i_k.get(key.s, key.click_s));
    }

    VecBasedMx dPji_dfjm = null; // dPji_dfjm(pi, fi) = dP(sesf.keys[pi]{ s, click_s -> e }) / df(sesf.keys[fi]{ s, click_s -> e })
    if (do_eval_gradient) {
      // 8 & для каждой пары блоков $i$, $j$ вычислить $\frac{d}{df}P(i,j)$
      dPji_dfjm = new VecBasedMx(sesf.f_count, sesf.f_count); // fPji_dfjm(pi, fi) = dP(sesf.keys[pi]{ s, click_s -> e }) / df(sesf.keys[fi]{ s, click_s -> e })
      for (int pi = 0; pi < sesf.f_count; pi++) {
        final double p_pi = P_i_j.get(pi);
        final double val2 = - p_pi * p_pi / f.get(pi);
        final SessionFeatureRepresentation.FeatureKey pi_key = sesf.keys.get(pi);
        for (final int j: ses.getEdgesFrom(pi_key.s)) {
          final int fi = sesf.keys_hash.get(new SessionFeatureRepresentation.FeatureKey(pi_key.s, j, pi_key.click_s));
          dPji_dfjm.set(pi, fi, val2);
        }
        dPji_dfjm.adjust(pi, pi, p_pi / f.get(pi));
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
      final VecBasedMx A = new VecBasedMx(blocks.length, MAX_PATH_LENGTH + 1);

      // Критерий ранней остановки вычисления матрицы $A$: если
      //   A(s,i,t+1) < \varepsilon \max_{i,u<=t} A(s,i,u)
      // то все пути длины $>t$ маловероятны по сравнению с путями длины $\le t$, и значения $A$ и $\frac{dA}{d\Theta}$ можно не вычислять
      int max_path_length_pruned = MAX_PATH_LENGTH;

      // A(s,i,1) =& P(s\to i) \cdot P(c=0 | i)              &\quad \forall i
      double max_A_lte_t = 0.;
      for (final int i: ses.getEdgesFrom(s)) {
        final double val = P_i_j.get(sesf.keys_hash.get(new SessionFeatureRepresentation.FeatureKey(s, i, click_s))) * P_noclick_i[i];
        A.set(i, 1, val);
        max_A_lte_t = Math.max(max_A_lte_t, val);
      }
      for (int t = 1; t < MAX_PATH_LENGTH; t++) {
        double max_A_tp1 = 0;
        for (int i = 0; i < blocks.length; i++) {
          double val = 0;
          for (final int j: ses.getEdgesTo(i))
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
      final double observation_prob = sumA_e_t * (1 - P_noclick_i[e]) / P_noclick_i[e];
      ret.loglikelihood += Math.log(observation_prob);

      if (do_eval_gradient) {
        // 9 & для всех блоков $i$ и длин $t$ вычислить $\frac{dA}{df_{fi}}(s,i,t)$
        final Tensor3 dA_df = new Tensor3(blocks.length, max_path_length_pruned + 1, sesf.f_count);
        // \frac{dA}{d\Theta}(s,i,1) =& P(c=0 | i) \cdot \frac{d}{d\Theta}P(s\to i) + P(s\to i) \cdot \frac{d}{d\Theta} P(c=0|i) \qquad \forall i
        for (final int i: ses.getEdgesFrom(s)) {
          final ArrayVec val = (ArrayVec)dPji_dfjm.row(sesf.keys_hash.get(new SessionFeatureRepresentation.FeatureKey(s, i, click_s)));
          val.scale(P_noclick_i[i]);
          dA_df.setRow(i, 1, val);
        }
        // \frac{dA}{df}(s,i,t+1) =&
        //  P(c=0|i) \sum_j \left( \frac{dA}{df}(s,j,t) \cdot P(j\to i)
        //       + A(s,j,t) \cdot \frac{d}{d\Theta} P(j\to i) \right)
        final ArrayVec sum = new ArrayVec(sesf.f_count);
        final ArrayVec val1 = new ArrayVec(sesf.f_count);
        final ArrayVec val2 = new ArrayVec(sesf.f_count);
        for (int t = 1; t < max_path_length_pruned; t++) {
          for (int i = 0; i < blocks.length; i++) {
            sum.fill(0);
            for (final int j: ses.getEdgesTo(i)) {
              final int f_ji0_index = sesf.keys_hash.get(new SessionFeatureRepresentation.FeatureKey(j, i, 0));
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
        final ArrayVec dPlogO_df = new ArrayVec(sesf.f_count);
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

  @Override
  public String explainTheta() {
    return "NOT IMPLEMENTED";
  }

  @Override
  public String toString() {
    return "NOT IMPLEMENTED";
  }

  public ClickProbabilityModel<Blk> getClickProbabilityModel() {
    return clickProbabilityModel;
  }

  public void setClickProbabilityModel(final ClickProbabilityModel<Blk> clickProbabilityModel) {
    this.clickProbabilityModel = clickProbabilityModel;
  }

  public AttractivenessModel<Blk> getAttractivenessModel() {
    return attractivenessModel;
  }

  public void setAttractivenessModel(final AttractivenessModel<Blk> attractivenessModel) {
    this.attractivenessModel = attractivenessModel;
  }

  @Override
  public double getClickGivenViewProbability(final Blk b) {
    return clickProbabilityModel.getClickGivenViewProbability(b);
  }

  @Override
  public double eval_f(final Session<Blk> ses, final int s, final int e, final int click_s) {
    return attractivenessModel.eval_f(ses, s, e, click_s);
  }

  @Override
  public SparseVec feats(final Session<Blk> ses, final int s, final int e, final int click_s) {
    return attractivenessModel.feats(ses, s, e, click_s);
  }

  @Override
  public int getEdgeFeatCount() {
    return attractivenessModel.getEdgeFeatCount();
  }
}
