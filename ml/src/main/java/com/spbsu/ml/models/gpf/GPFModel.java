package com.spbsu.ml.models.gpf;

import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;

import java.util.ArrayList;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: irlab
 * Date: 08.07.14
 * Time: 13:31
 * To change this template use File | Settings | File Templates.
 */
public interface GPFModel {
  /**
   * this function sets up a structure of a Session: a set of vertices (blocks and virtual blocks), and a set of edges
   * @param ses - a Session to set up (write-only)
   * @param result_blocks - a set of 'real' (observed) blocks (read-only)
   * @param clicks_block_indexes - list of clicks (clicks_block_indexes[i] is a i'th click on result_blocks[clicks_block_indexes[i]])
   */
  void setSessionData(Session ses, Session.Block[] result_blocks, int[] clicks_block_indexes);

  double eval_f(Session ses, int s, int e, int click_s);

  String explainTheta();

  String explainSessionProb(Session ses);

  double getClickGivenViewProbability(Session.Block b);

  /**
   * @param ses - session
   * @return transmx_0[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
   */
  VecBasedMx evalSessionTransitionProbs(Session ses);

  /**
   * for each block in ses.getBlocks(), evaluates probability that a user has one or more clicks on the block, given the behavior model
   * @param ses - viewport structure
   * @return double[ses.getBlocks().length] - array of probabilities
   */
  double[] evalHasClickProbabilities(Session ses);

  /**
   * for each block in ses.getBlocks(), evaluates probability that a user has one or more views on the block, given the behavior model
   * @param ses - viewport structure
   * @return double[ses.getBlocks().length] - array of probabilities
   */
  double[] evalHasViewProbabilities(Session ses);

  /**
   * for each block in ses.getBlocks(), evaluates expected number of steps when a user looks at the block, given the behavior model
   * this is not distribution, sum of values is not equal to 1
   * @param ses - viewport structure
   * @return double[ses.getBlocks().length] - array of expected number of steps
   */
  double[] evalExpectedAttention(Session ses);

  abstract class Stub implements GPFModel {
    public int MAX_PATH_LENGTH = 15;

    /**
     * this function sets up the structure of a Session: a set of vertices (blocks and virtual blocks), and a set of edges
     * @param ses - a Session to set up (write-only)
     * @param result_blocks - a set of 'real' (observed) blocks (read-only)
     * @param clicks_block_indexes - list of clicks (clicks_block_indexes[i] is a i'th click on result_blocks[clicks_block_indexes[i]])
     */
    public void setSessionData(Session ses, Session.Block[] result_blocks, int[] clicks_block_indexes) {
      // init blocks
      Session.Block[] blocks = new Session.Block[result_blocks.length + Session.R0_ind];
//    int[] result_pos2block_ind = new int[100];
      int max_result_pos = -1;
      int min_result_pos = 1000;

      blocks[Session.Q_ind] = new Session.Block(Session.BlockType.Q, null, -1, null);
      blocks[Session.S_ind] = new Session.Block(Session.BlockType.S, null, -1, null);
      blocks[Session.E_ind] = new Session.Block(Session.BlockType.E, null, -1, null);
      for (int i = 0; i < result_blocks.length; i++) {
        blocks[i + Session.R0_ind] = result_blocks[i];
        max_result_pos = Math.max(max_result_pos, result_blocks[i].position);
        min_result_pos = Math.min(min_result_pos, result_blocks[i].position);
      }
      ses.setBlocks(blocks);

      int[] click_indexes = new int[clicks_block_indexes.length];
      for (int i = 0; i < click_indexes.length; i++)
        click_indexes[i] = clicks_block_indexes[i] + Session.R0_ind;
      ses.setClick_indexes(click_indexes);

      // init edges
      List<Session.Edge> edges = new ArrayList<Session.Edge>();
      for (int i = Session.R0_ind; i < blocks.length; i++) {
        // R_i -> R_{i+1}
        if (i + 1 < blocks.length)
          edges.add(new Session.Edge(i, i+1));
        // R_i -> R_{i-1}
        if (i > Session.R0_ind)
          edges.add(new Session.Edge(i, i-1));
        // Q -> R_i
        edges.add(new Session.Edge(Session.Q_ind, i));
        // S -> R_i
        edges.add(new Session.Edge(Session.S_ind, i));
        // R_i -> S
        edges.add(new Session.Edge(i, Session.S_ind));
        // R_i -> E
        edges.add(new Session.Edge(i, Session.E_ind));
        // E -> E
        edges.add(new Session.Edge(Session.E_ind, Session.E_ind));
      }
      ses.setEdges(edges);
    }

    /**
     * @param ses - session
     * @return transmx_0[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
     */
    public VecBasedMx evalSessionTransitionProbs(Session ses) {
      final Session.Block[] blocks = ses.getBlocks();

      // 1 & для каждой пары блоков $i$, $j$ вычислить $f(i,j)$; третья координата - наличие клика c_i
      Tensor3 f = new Tensor3(blocks.length, blocks.length, 2);
      for (int i = 0; i < blocks.length; i++) {
        for (int j: ses.getEdgesFrom(i)) {
          for (int click_i = 0; click_i < 2; click_i++) {
            f.set(i, j, click_i, eval_f(ses, i, j, click_i));
          }
        }
      }
      f.set(Session.E_ind, Session.E_ind, 0, 1.);
      f.set(Session.E_ind, Session.E_ind, 1, 1.);

      // transmx[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
      VecBasedMx transmx_0 = new VecBasedMx(blocks.length * 2, blocks.length * 2);
      for (int i = 0; i < blocks.length; i++) {
        for (int click_i = 0; click_i < 2; click_i++) {
          double sum_f_i_j = 0.;
          for (int j: ses.getEdgesFrom(i))
            sum_f_i_j += f.get(i, j, click_i);

          for (int j: ses.getEdgesFrom(i)) {
            double trans_prob = f.get(i, j, click_i) / sum_f_i_j;
            double click_prob = getClickGivenViewProbability(blocks[j]);
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
    public double[] evalHasClickProbabilities(Session ses) {
      final Session.Block[] blocks = ses.getBlocks();
      // transmx_0[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
      VecBasedMx transmx_0 = evalSessionTransitionProbs(ses);

      double[] hasClickProbabilities = new double[blocks.length];
      for (int ci = Session.R0_ind; ci < blocks.length; ci++) {
        VecBasedMx transmx_ci = new VecBasedMx(transmx_0);
        // модифицируем стохастическую матрицу transmx_ci так, чтобы после клика на ci пользователь оставался в том же состоянии
        for (int j = 0; j < transmx_ci.columns; j++)
          transmx_ci.set(1 * blocks.length + ci, j, 0.);
        transmx_ci.set(1 * blocks.length + ci, 1 * blocks.length + ci, 1.);

        // сначала пользователь в состоянии (Q, no_click)
        Vec state_probabilities = new ArrayVec(transmx_ci.columns);
        state_probabilities.set(Session.Q_ind, 1.);

        for (int t = 0; t < MAX_PATH_LENGTH; t++)
          state_probabilities = MxTools.multiply(transmx_ci, state_probabilities);

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
    public double[] evalHasViewProbabilities(Session ses) {
      final Session.Block[] blocks = ses.getBlocks();
      // transmx_0[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
      VecBasedMx transmx_0 = evalSessionTransitionProbs(ses);

      double[] hasViewProbabilities = new double[blocks.length];
      for (int ci = Session.R0_ind; ci < blocks.length; ci++) {
        VecBasedMx transmx_ci = new VecBasedMx(transmx_0);
        // модифицируем стохастическую матрицу transmx_ci так, чтобы после view на ci пользователь оставался в том же состоянии
        for (int j = 0; j < transmx_ci.columns; j++) {
          transmx_ci.set(0 * blocks.length + ci, j, 0.);
          transmx_ci.set(1 * blocks.length + ci, j, 0.);
        }
        transmx_ci.set(0 * blocks.length + ci, 0 * blocks.length + ci, 1.);
        transmx_ci.set(1 * blocks.length + ci, 1 * blocks.length + ci, 1.);

        // сначала пользователь в состоянии (Q, no_click)
        Vec state_probabilities = new ArrayVec(transmx_ci.columns);
        state_probabilities.set(Session.Q_ind, 1.);

        for (int t = 0; t < MAX_PATH_LENGTH; t++)
          state_probabilities = MxTools.multiply(transmx_ci, state_probabilities);

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
    public double[] evalExpectedAttention(Session ses) {
      final Session.Block[] blocks = ses.getBlocks();
      // transmx_0[(i, click_i), (j, click_j)] - вероятность перехода за один шаг из состояния (i, click_i) в (j, click_j)
      VecBasedMx transmx_0 = evalSessionTransitionProbs(ses);

      double[] expectedAttention = new double[blocks.length];
      // сначала пользователь в состоянии (Q, no_click)
      Vec state_probabilities = new ArrayVec(transmx_0.columns);
      state_probabilities.set(Session.Q_ind, 1.);

      for (int t = 0; t < MAX_PATH_LENGTH; t++) {
        state_probabilities = MxTools.multiply(transmx_0, state_probabilities);
        for (int i = Session.R0_ind; i < blocks.length; i++)
          expectedAttention[i] += state_probabilities.get(0 * blocks.length + i) + state_probabilities.get(1 * blocks.length + i);
      }

      return expectedAttention;
    }

    public String explainSessionProb(Session ses) {
      VecBasedMx sum_f = new VecBasedMx(ses.getBlocks().length, 2);
      for (int i = 0; i < ses.getBlocks().length; i++) {
        for (int j: ses.getEdgesFrom(i)) {
          sum_f.adjust(i, 0, eval_f(ses, i, j, 0));
          sum_f.adjust(i, 1, eval_f(ses, i, j, 1));
        }
      }
      double[] hasClickProbabilities = evalHasClickProbabilities(ses);
      double[] hasViewProbabilities = evalHasViewProbabilities(ses);
      double[] attentionExpectation = evalExpectedAttention(ses);
//    ArrayVec attentionDistribution = new ArrayVec(attentionExpectation);
//    attentionDistribution.scale(1. / VecTools.sum(attentionDistribution));

      StringBuffer ret = new StringBuffer();
      ret.append("pos\tsntype\trel\tclick\t");
      ret.append("P(has_click)\tP(has_view)\tE(Att)\t\t");
      ret.append("P(click|V)\tP(Q->i)\tP(S->i)\t\t");
      ret.append("P(i->i+1|c=0)\tP(i->i-1|c=0)\tP(i->E|c=0)\tP(i->S|c=0)\t\t");
      ret.append("P(i->i+1|c=1)\tP(i->i-1|c=1)\tP(i->E|c=1)\tP(i->S|c=1)\n");

      for (int i = Session.R0_ind; i < ses.getBlocks().length; i++) {
        Session.Block bi = ses.getBlock(i);
        int click_position = -1;
        for (int ci = 0; ci < ses.getClick_indexes().length; ci++) {
          if (ses.getClick_indexes()[ci] == i) {
            click_position = ci+1;
            break;
          }
        }
        //ret.append("pos\tsntype\trel\tclick\t");
        ret.append("" + bi.position + "\t" + bi.resultType.name() + "\t" + bi.resultGrade.name() + "\t" + (click_position >= 0 ? click_position : "-") + "\t");

        //ret.append("P(has_click)\tP(has_view)\tAtt\t\t");
        ret.append("" + hasClickProbabilities[i] + "\t" + hasViewProbabilities[i] + "\t" + attentionExpectation[i] + "\t\t");

        //ret.append("P(click|V)\tP(Q->i)\tP(S->i)\t\t");
        double P_Q_i = eval_f(ses, Session.Q_ind, i, 0) / sum_f.get(Session.Q_ind, 0);
        double P_S_i = eval_f(ses, Session.S_ind, i, 0) / sum_f.get(Session.S_ind, 0);
        ret.append("" + getClickGivenViewProbability(bi) + "\t" + P_Q_i + "\t" + P_S_i + "\t\t");

        //ret.append("P(i->i+1|c=0)\tP(i->i-1|c=0)\tP(i->E|c=0)\tP(i->S|c=0)\t\t");
        int click_i = 0;
        double P_down = i + 1 < ses.getBlocks().length ? eval_f(ses, i, i+1, click_i) / sum_f.get(i, click_i) : 0;
        double P_up = i - 1 >= Session.R0_ind ? eval_f(ses, i, i-1, click_i) / sum_f.get(i, click_i) : 0;
        double P_i_E = eval_f(ses, i, Session.E_ind, click_i) / sum_f.get(i, click_i);
        double P_i_S = eval_f(ses, i, Session.S_ind, click_i) / sum_f.get(i, click_i);
        ret.append("" + P_down + "\t" + P_up + "\t" + P_i_E + "\t" + P_i_S + "\t\t");

        //ret.append("P(i->i+1|c=1)\tP(i->i-1|c=1)\tP(i->E|c=1)\tP(i->S|c=1)\n");
        click_i = 1;
        P_down = i + 1 < ses.getBlocks().length ? eval_f(ses, i, i+1, click_i) / sum_f.get(i, click_i) : 0;
        P_up = i - 1 >= Session.R0_ind ? eval_f(ses, i, i-1, click_i) / sum_f.get(i, click_i) : 0;
        P_i_E = eval_f(ses, i, Session.E_ind, click_i) / sum_f.get(i, click_i);
        P_i_S = eval_f(ses, i, Session.S_ind, click_i) / sum_f.get(i, click_i);
        ret.append("" + P_down + "\t" + P_up + "\t" + P_i_E + "\t" + P_i_S + "\n");
      }
      return ret.toString();
    }

  }
}
