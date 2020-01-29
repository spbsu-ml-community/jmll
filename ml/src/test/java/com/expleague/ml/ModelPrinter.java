package com.expleague.ml;

import com.expleague.commons.math.Trans;
import com.expleague.ml.func.Ensemble;

public class ModelPrinter implements ProgressHandler {
  @Override
  public void accept(final Trans partial) {
    if (partial instanceof Ensemble) {
      final Ensemble model = (Ensemble) partial;
      final Trans increment = model.last();
      System.out.print("\t" + increment);
    }
  }
}
