package com.spbsu.ml.models.gpf;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.ml.models.gpf.weblogmodel.BlockV1;

/**
 * Created with IntelliJ IDEA.
 * User: irlab
 * Date: 08.07.14
 * Time: 13:31
 * To change this template use File | Settings | File Templates.
 */
public interface GPFModel<Blk extends Session.Block> extends AttractivenessModel<Blk> {
  String explainTheta();

  String explainSessionProb(Session<Blk> ses);

  double getClickGivenViewProbability(Blk b);

  /**
   * @param ses - session
   * @return transmx_0[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
   */
  VecBasedMx evalSessionTransitionProbs(Session<Blk> ses);

  /**
   * for each block in ses.getBlocks(), evaluates probability that a user has one or more clicks on the block, given the behavior model
   * @param ses - viewport structure
   * @return double[ses.getBlocks().length] - array of probabilities
   */
  double[] evalHasClickProbabilities(Session<Blk> ses);

  /**
   * for each block in ses.getBlocks(), evaluates probability that a user has one or more views on the block, given the behavior model
   * @param ses - viewport structure
   * @return double[ses.getBlocks().length] - array of probabilities
   */
  double[] evalHasViewProbabilities(Session<Blk> ses);

  /**
   * for each block in ses.getBlocks(), evaluates expected number of steps when a user looks at the block, given the behavior model
   * this is not distribution, sum of values is not equal to 1
   * @param ses - viewport structure
   * @return double[ses.getBlocks().length] - array of expected number of steps
   */
  double[] evalExpectedAttention(Session<Blk> ses);

  /**
   * evaluates expected number of clicks on the SERP, given the behavior model
   * @param ses - viewport structure
   * @return double - array of expected number of steps
   */
  double evalExpectedNumberOfClicks(Session<Blk> ses);

  abstract class Stub<Blk extends Session.Block> implements GPFModel<Blk> {
    public int MAX_PATH_LENGTH = 15;

    /**
     * @param ses - session
     * @return transmx_0[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
     */
    @Override
    public VecBasedMx evalSessionTransitionProbs(final Session<Blk> ses) {
      final Blk[] blocks = ses.getBlocks();

      // 1 & для каждой пары блоков $i$, $j$ вычислить $f(i,j)$; третья координата - наличие клика c_i
      final Tensor3 f = new Tensor3(blocks.length, blocks.length, 2);
      for (int i = 0; i < blocks.length; i++) {
        for (final int j: ses.getEdgesFrom(i)) {
          for (int click_i = 0; click_i < 2; click_i++) {
            f.set(i, j, click_i, eval_f(ses, i, j, click_i));
          }
        }
      }
      f.set(Session.E_INDEX, Session.E_INDEX, 0, 1.);
      f.set(Session.E_INDEX, Session.E_INDEX, 1, 1.);

      // transmx[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
      final VecBasedMx transmx_0 = new VecBasedMx(blocks.length * 2, blocks.length * 2);
      for (int i = 0; i < blocks.length; i++) {
        for (int click_i = 0; click_i < 2; click_i++) {
          double sum_f_i_j = 0.;
          for (final int j: ses.getEdgesFrom(i))
            sum_f_i_j += f.get(i, j, click_i);

          for (final int j: ses.getEdgesFrom(i)) {
            final double trans_prob = f.get(i, j, click_i) / sum_f_i_j;
            final double click_prob = getClickGivenViewProbability(blocks[j]);
            transmx_0.set(click_i * blocks.length + i, 0 * blocks.length + j, trans_prob * (1. - click_prob));
            transmx_0.set(click_i * blocks.length + i, 1 * blocks.length + j, trans_prob * click_prob);
          }
        }
      }
      return transmx_0;
    }

    /**
     * for each block in ses.getBlocks(), evaluates probability that a user has one or more clicks on the block, given the behavior model
     * @param ses - viewport structure
     * @return double[ses.getBlocks().length] - array of probabilities
     */
    @Override
    public double[] evalHasClickProbabilities(final Session<Blk> ses) {
      final Session.Block[] blocks = ses.getBlocks();
      // transmx_0[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
      final VecBasedMx transmx_0 = evalSessionTransitionProbs(ses);

      final double[] hasClickProbabilities = new double[blocks.length];
      for (int ci = Session.R0_INDEX; ci < blocks.length; ci++) {
        final VecBasedMx transmx_ci = new VecBasedMx(transmx_0);
        // модифицируем стохастическую матрицу transmx_ci так, чтобы после клика на ci пользователь оставался в том же состоянии
        for (int j = 0; j < transmx_ci.columns; j++)
          transmx_ci.set(1 * blocks.length + ci, j, 0.);
        transmx_ci.set(1 * blocks.length + ci, 1 * blocks.length + ci, 1.);

        // сначала пользователь в состоянии (Q, no_click)
        Mx state_probabilities = new VecBasedMx(1, transmx_0.columns);
        state_probabilities.set(0, Session.Q_INDEX, 1.);

        for (int t = 0; t < MAX_PATH_LENGTH; t++)
          state_probabilities = MxTools.multiply(state_probabilities, transmx_ci);

        // вероятность через MAX_PATH_LENGTH шагов остаться в состоянии (ci, click)
        hasClickProbabilities[ci] = state_probabilities.get(1 * blocks.length + ci);
      }

      return hasClickProbabilities;
    }

    /**
     * for each block in ses.getBlocks(), evaluates probability that a user has one or more views on the block, given the behavior model
     * @param ses - viewport structure
     * @return double[ses.getBlocks().length] - array of probabilities
     */
    @Override
    public double[] evalHasViewProbabilities(final Session<Blk> ses) {
      final Session.Block[] blocks = ses.getBlocks();
      // transmx_0[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
      final VecBasedMx transmx_0 = evalSessionTransitionProbs(ses);

      final double[] hasViewProbabilities = new double[blocks.length];
      for (int ci = Session.R0_INDEX; ci < blocks.length; ci++) {
        final VecBasedMx transmx_ci = new VecBasedMx(transmx_0);
        // модифицируем стохастическую матрицу transmx_ci так, чтобы после view на ci пользователь оставался в том же состоянии
        for (int j = 0; j < transmx_ci.columns; j++) {
          transmx_ci.set(0 * blocks.length + ci, j, 0.);
          transmx_ci.set(1 * blocks.length + ci, j, 0.);
        }
        transmx_ci.set(0 * blocks.length + ci, 0 * blocks.length + ci, 1.);
        transmx_ci.set(1 * blocks.length + ci, 1 * blocks.length + ci, 1.);

        // сначала пользователь в состоянии (Q, no_click)
        Mx state_probabilities = new VecBasedMx(1, transmx_0.columns);
        state_probabilities.set(0, Session.Q_INDEX, 1.);

        for (int t = 0; t < MAX_PATH_LENGTH; t++)
          state_probabilities = MxTools.multiply(state_probabilities, transmx_ci);

        // вероятность через MAX_PATH_LENGTH шагов остаться в состоянии (ci, click) или (ci, noclick)
        hasViewProbabilities[ci] = state_probabilities.get(0 * blocks.length + ci) + state_probabilities.get(1 * blocks.length + ci);
      }

      return hasViewProbabilities;
    }

    /**
     * for each block in ses.getBlocks(), evaluates expected number of steps when a user looks at the block, given the behavior model
     * this is not distribution, sum of values is not equal to 1
     * @param ses - viewport structure
     * @return double[ses.getBlocks().length] - array of expected number of steps
     */
    @Override
    public double[] evalExpectedAttention(final Session<Blk> ses) {
      final Session.Block[] blocks = ses.getBlocks();
      // transmx_0[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
      final VecBasedMx transmx_0 = evalSessionTransitionProbs(ses);

      final double[] expectedAttention = new double[blocks.length];
      // сначала пользователь в состоянии (Q, no_click)
      Mx state_probabilities = new VecBasedMx(1, transmx_0.columns);
      state_probabilities.set(0, Session.Q_INDEX, 1.);

      for (int t = 0; t < MAX_PATH_LENGTH; t++) {
        state_probabilities = MxTools.multiply(state_probabilities, transmx_0);
        for (int i = Session.R0_INDEX; i < blocks.length; i++)
          expectedAttention[i] += state_probabilities.get(0 * blocks.length + i) + state_probabilities.get(1 * blocks.length + i);
      }

      return expectedAttention;
    }

    /**
     * for each block in ses.getBlocks(), evaluates expected number of steps when a user looks at the block, given the behavior model
     * this is not distribution, sum of values is not equal to 1
     * @param ses - viewport structure
     * @return double[ses.getBlocks().length] - array of expected number of steps
     */
    @Override
    public double evalExpectedNumberOfClicks(final Session<Blk> ses) {
      final Session.Block[] blocks = ses.getBlocks();
      // transmx_0[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
      final VecBasedMx transmx_0 = evalSessionTransitionProbs(ses);

      double expectedNumberOfClicks = 0.;
      // сначала пользователь в состоянии (Q, no_click)
      Mx state_probabilities = new VecBasedMx(1, transmx_0.columns);
      state_probabilities.set(0, Session.Q_INDEX, 1.);

      for (int t = 0; t < MAX_PATH_LENGTH; t++) {
        state_probabilities = MxTools.multiply(state_probabilities, transmx_0);
        for (int i = Session.R0_INDEX; i < blocks.length; i++)
          expectedNumberOfClicks += state_probabilities.get(blocks.length + i);
      }

      return expectedNumberOfClicks;
    }

    @Override
    public String explainSessionProb(final Session<Blk> ses) {
      final VecBasedMx sum_f = new VecBasedMx(ses.getBlocks().length, 2);
      for (int i = 0; i < ses.getBlocks().length; i++) {
        for (final int j: ses.getEdgesFrom(i)) {
          sum_f.adjust(i, 0, eval_f(ses, i, j, 0));
          sum_f.adjust(i, 1, eval_f(ses, i, j, 1));
        }
      }
      final double[] hasClickProbabilities = evalHasClickProbabilities(ses);
      final double[] hasViewProbabilities = evalHasViewProbabilities(ses);
      final double[] attentionExpectation = evalExpectedAttention(ses);
//    ArrayVec attentionDistribution = new ArrayVec(attentionExpectation);
//    attentionDistribution.scale(1. / VecTools.sum(attentionDistribution));

      final StringBuffer ret = new StringBuffer();
      ret.append("pos\tsntype\trel\tclick\t");
      ret.append("P(has_click)\tP(has_view)\tE(Att)\t\t");
      ret.append("P(click|V)\tP(Q->i)\tP(S->i)\t\t");
      ret.append("P(i->i+1|c=0)\tP(i->i-1|c=0)\tP(i->E|c=0)\tP(i->S|c=0)\t\t");
      ret.append("P(i->i+1|c=1)\tP(i->i-1|c=1)\tP(i->E|c=1)\tP(i->S|c=1)\n");

      for (int i = Session.R0_INDEX; i < ses.getBlocks().length; i++) {
        final Blk bi = ses.getBlock(i);
        int click_position = -1;
        for (int ci = 0; ci < ses.getClick_indexes().length; ci++) {
          if (ses.getClick_indexes()[ci] == i) {
            click_position = ci+1;
            break;
          }
        }
        //ret.append("pos\tsntype\trel\tclick\t");
        ret.append("" + bi.position + "\t" +
                   (bi instanceof BlockV1 ? ((BlockV1)bi).resultType.name() : "?") + "\t" +
                   (bi instanceof BlockV1 ? ((BlockV1)bi).resultGrade.name() : "?") + "\t" +
                   (click_position >= 0 ? click_position : "-") + "\t");

        //ret.append("P(has_click)\tP(has_view)\tAtt\t\t");
        ret.append("" + hasClickProbabilities[i] + "\t" + hasViewProbabilities[i] + "\t" + attentionExpectation[i] + "\t\t");

        //ret.append("P(click|V)\tP(Q->i)\tP(S->i)\t\t");
        final double P_Q_i = eval_f(ses, Session.Q_INDEX, i, 0) / sum_f.get(Session.Q_INDEX, 0);
        final double P_S_i = eval_f(ses, Session.S_INDEX, i, 0) / sum_f.get(Session.S_INDEX, 0);
        ret.append("" + getClickGivenViewProbability(bi) + "\t" + P_Q_i + "\t" + P_S_i + "\t\t");

        //ret.append("P(i->i+1|c=0)\tP(i->i-1|c=0)\tP(i->E|c=0)\tP(i->S|c=0)\t\t");
        int click_i = 0;
        double P_down = i + 1 < ses.getBlocks().length ? eval_f(ses, i, i+1, click_i) / sum_f.get(i, click_i) : 0;
        double P_up = i - 1 >= Session.R0_INDEX ? eval_f(ses, i, i-1, click_i) / sum_f.get(i, click_i) : 0;
        double P_i_E = eval_f(ses, i, Session.E_INDEX, click_i) / sum_f.get(i, click_i);
        double P_i_S = eval_f(ses, i, Session.S_INDEX, click_i) / sum_f.get(i, click_i);
        ret.append("" + P_down + "\t" + P_up + "\t" + P_i_E + "\t" + P_i_S + "\t\t");

        //ret.append("P(i->i+1|c=1)\tP(i->i-1|c=1)\tP(i->E|c=1)\tP(i->S|c=1)\n");
        click_i = 1;
        P_down = i + 1 < ses.getBlocks().length ? eval_f(ses, i, i+1, click_i) / sum_f.get(i, click_i) : 0;
        P_up = i - 1 >= Session.R0_INDEX ? eval_f(ses, i, i-1, click_i) / sum_f.get(i, click_i) : 0;
        P_i_E = eval_f(ses, i, Session.E_INDEX, click_i) / sum_f.get(i, click_i);
        P_i_S = eval_f(ses, i, Session.S_INDEX, click_i) / sum_f.get(i, click_i);
        ret.append("" + P_down + "\t" + P_up + "\t" + P_i_E + "\t" + P_i_S + "\n");
      }
      return ret.toString();
    }

  }
}
