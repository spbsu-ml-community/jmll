package com.spbsu.ml.methods.wrappers;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.WeakListenerHolder;
import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.ml.TargetFunc;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.MultiLabelTools;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.multilabel.MultiLabelModel;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;

/**
 * User: qdeee
 * Date: 03.04.15
 */
public class MultiLabelWrapper<GlobalLoss extends TargetFunc> extends WeakListenerHolderImpl<Trans> implements VecOptimization<GlobalLoss> {
  private final VecOptimization<GlobalLoss> strong;

  public MultiLabelWrapper(final VecOptimization<GlobalLoss> strong) {
    this.strong = strong;
  }

  @Override
  public MultiLabelModel fit(final VecDataSet learn, final GlobalLoss targetFunc) {
    List<Action> internListeners = new ArrayList<>();
    if (strong instanceof WeakListenerHolder) {
      for (WeakReference<Action<? super Trans>> externalListenerRef : listeners) {
        final Action<? super Trans> externalListener = externalListenerRef.get();
        if (externalListener != null) {
          final Action<Trans> internListener = new Action<Trans>() {
            @Override
            public void invoke(final Trans trans) {
              externalListener.invoke(trans);
            }
          };
          internListeners.add(internListener);
          ((WeakListenerHolder) strong).addListener(internListener);
        }
      }
    }

    final Trans model = strong.fit(learn, targetFunc);
    return MultiLabelTools.extractMultiLabelModel(model);
  }
}
