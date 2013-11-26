package com.spbsu.ml.methods;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.CompositeFunc;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.DataSet;

/**
* User: solar
* Date: 21.12.2010
* Time: 22:13:54
*/
public abstract class CompositeOptimization<GlobalLoss extends Func, LocalLoss extends Func> extends WeakListenerHolderImpl<Func> implements Optimization<GlobalLoss> {
  private static final Logger LOG = Logger.create(CompositeOptimization.class);
  protected final Optimization<LocalLoss> weak;
  int iterationsCount;

  public CompositeOptimization(Optimization<LocalLoss> weak, int iterationsCount) {
    this.weak = weak;
    this.iterationsCount = iterationsCount;
  }

  public CompositeFunc fit(DataSet learn, GlobalLoss loss) {
    LOG.assertTrue(learn.power() == loss.xdim(), "Target must fit the loss dimension");
    CompositeFunc result = null;
    for (int t = 0; t < iterationsCount; t++) {
      final LocalLoss[] losses = transferAt(loss, result);
      for (int i = 0; i < losses.length; i++) {
        final Func weak = this.weak.fit(learn, losses[i]);
        result = combine(result, weak, i);
      }
      invoke(result);
    }
    return result;
  }

  public abstract LocalLoss[] transferAt(GlobalLoss local, Func currentResult);
  public abstract CompositeFunc combine(CompositeFunc current, Func increment, int dim);
}
