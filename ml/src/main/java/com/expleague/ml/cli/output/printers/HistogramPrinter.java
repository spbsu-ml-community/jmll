package com.expleague.ml.cli.output.printers;

import com.expleague.ml.ProgressHandler;
import com.expleague.commons.math.Trans;
import com.expleague.ml.dynamicGrid.models.ObliviousTreeDynamicBin;
import com.expleague.ml.func.Ensemble;

import java.util.Arrays;

/**
 * User: qdeee
 * Date: 04.09.14
 */
public class HistogramPrinter implements ProgressHandler {
  int iteration = 0;

  @Override
  public void accept(final Trans partial) {
    iteration++;
    if (iteration % 10 != 0) {
      return;
    }
    if (partial instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) partial;
      final double step = ensemble.wlast();
      final Trans last = ensemble.last();
      if (last instanceof ObliviousTreeDynamicBin) {
        final ObliviousTreeDynamicBin tree = (ObliviousTreeDynamicBin) last;
        System.out.println("Current grid " + Arrays.toString(tree.grid().hist()));
      }
    }
  }
}
