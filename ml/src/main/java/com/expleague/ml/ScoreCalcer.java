package com.expleague.ml;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.func.Ensemble;

public class ScoreCalcer implements ProgressHandler {
  final String message;
  final Vec current;
  private final VecDataSet ds;
  private final TargetFunc target;
  private final boolean minimize;
  private int period;

  public ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target) {
    this(message, ds, target, true, 1);
  }

  public ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target, boolean minimize, int period) {
    this.message = message;
    this.ds = ds;
    this.target = target;
    current = new ArrayVec(ds.length());
    best = minimize ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
    this.minimize = minimize;
    this.period = period;
  }

  double best;

  int iter = 0;
  @Override
  public void accept(final Trans partial) {
    if (partial instanceof Ensemble) {
      final Ensemble linear = (Ensemble) partial;
      final Trans increment = linear.last();
      for (int i = 0; i < ds.length(); i++) {
        if (increment instanceof Ensemble) {
          current.adjust(i, linear.wlast() * (increment.trans(ds.data().row(i)).get(0)));
        } else {
          current.adjust(i, linear.wlast() * ((Func) increment).value(ds.data().row(i)));
        }
      }
    } else {
      for (int i = 0; i < ds.length(); i++) {
        current.set(i, ((Func) partial).value(ds.data().row(i)));
      }
    }
    if (++iter % period == 0) {
      final double value = target.value(current);
      System.out.print(message + value);
      best = minimize ? Math.min(value, best) : Math.max(value, best);
      System.out.print(" best = " + best);
    }
  }
}
