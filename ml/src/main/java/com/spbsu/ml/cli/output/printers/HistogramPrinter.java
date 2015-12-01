package com.spbsu.ml.cli.output.printers;

import com.spbsu.ml.ProgressHandler;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.dynamicGrid.models.ObliviousTreeDynamicBin;
import com.spbsu.ml.func.Ensemble;

import java.util.Arrays;

/**
 * User: qdeee
 * Date: 04.09.14
 */
public class HistogramPrinter implements ProgressHandler {
  int iteration = 0;

  @Override
  public void invoke(final Trans partial) {
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
