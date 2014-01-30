package com.spbsu.ml.methods;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.models.ProbabilisticGraphicalModel;

/**
 * User: solar
 * Date: 27.01.14
 * Time: 13:29
 */
public class PGMEM implements Optimization<LLLogit> {
  private final Mx topology;
  private final int iterations;
  private double step;

  public PGMEM(Mx topology, double smoothing, int iterations) {
    this.topology = topology;
    this.iterations = iterations;
    this.step = smoothing;
  }

  @Override
  public ProbabilisticGraphicalModel fit(DataSet learn, LLLogit ll) {
    ProbabilisticGraphicalModel currentPGM = new ProbabilisticGraphicalModel(topology);
    int[][] cpds = new int[learn.power()][];
    final Mx data = learn.data();
    for (int j = 0; j < data.rows(); j++) {
      cpds[j] = currentPGM.extractControlPoints(data.row(j));
    }
    final ProbabilisticGraphicalModel.Route[] eroutes = new ProbabilisticGraphicalModel.Route[learn.power()];
    for (int t = 0; t < iterations; t++) {
      { // E-step
        for (int j = 0; j < cpds.length; j++) {
          final int finalJ = j;
          currentPGM.visit(new Action<ProbabilisticGraphicalModel.Route>() {
            @Override
            public void invoke(ProbabilisticGraphicalModel.Route route) {
              if (eroutes[finalJ].probab < route.probab)
                eroutes[finalJ] = route;
            }
          }, cpds[j]);
        }
      }
      final Mx next = new VecBasedMx(topology.columns(), new ArrayVec(topology.dim()));
      VecTools.fill(next, 1.); // adjusting parameters of Dir(next[i]) by one
      { // M-step
        for (ProbabilisticGraphicalModel.Route eroute : eroutes) {
          byte prev = eroute.nodes[0];
          for (int i = 1; i < eroute.nodes.length; i++) {
            next.adjust(prev, prev = eroute.nodes[i], 1.);
          }
        }
        for (int i = 0; i < next.rows(); i++) {
          VecTools.normalizeL1(next.row(i)); // assuming weights of nodes are distributed by Dir(next[i]), then optimal parameters will be proportional to pass count
        }
      }
      { // Update PGM
        VecTools.scale(next, step/(1. - step));
        VecTools.append(next, currentPGM.topology);
        VecTools.scale(next, (1. - step));
        currentPGM = new ProbabilisticGraphicalModel(next);
      }
    }
    return currentPGM;
  }
}
