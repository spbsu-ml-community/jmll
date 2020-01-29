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

  public ScoreCalcer(final String message, final VecDataSet ds, final TargetFunc target) {
    this.message = message;
    this.ds = ds;
    this.target = target;
    current = new ArrayVec(ds.length());
  }

  double min = 1e10;

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
    final double value = target.value(current);
    System.out.print(message + value);
    min = Math.min(value, min);
    System.out.print(" best = " + min);
  }
}
