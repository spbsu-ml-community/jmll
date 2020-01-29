package com.expleague.ml;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.L2;
import gnu.trove.map.hash.TDoubleDoubleHashMap;
import gnu.trove.map.hash.TDoubleIntHashMap;

import static com.expleague.commons.math.MathTools.sqr;
import static java.lang.Math.log;

public class QualityCalcer implements ProgressHandler {
  Vec residues = VecTools.copy(GridTest.learn.target(L2.class).target);
  double total = 0;
  int index = 0;

  @Override
  public void accept(final Trans partial) {
    if (partial instanceof Ensemble) {
      final Ensemble model = (Ensemble) partial;
      final Trans increment = model.last();

      final TDoubleIntHashMap values = new TDoubleIntHashMap();
      final TDoubleDoubleHashMap dispersionDiff = new TDoubleDoubleHashMap();
      int index = 0;
      final VecDataSet ds = GridTest.learn.vecData();
      for (int i = 0; i < ds.data().rows(); i++) {
        final double value;
        if (increment instanceof Ensemble) {
          value = increment.trans(ds.data().row(i)).get(0);
        } else {
          value = ((Func) increment).value(ds.data().row(i));
        }
        values.adjustOrPutValue(value, 1, 1);
        final double ddiff = sqr(residues.get(index)) - sqr(residues.get(index) - value);
        residues.adjust(index, -model.wlast() * value);
        dispersionDiff.adjustOrPutValue(value, ddiff, ddiff);
        index++;
      }
//          double totalDispersion = VecTools.multiply(residues, residues);
      double score = 0;
      for (final double key : values.keys()) {
        final double regularizer = 1 - 2 * log(2) / log(values.get(key) + 1);
        score += dispersionDiff.get(key) * regularizer;
      }
//          score /= totalDispersion;
      total += score;
      this.index++;
      System.out.print("\tscore:\t" + score + "\tmean:\t" + (total / this.index));
    }
  }
}
