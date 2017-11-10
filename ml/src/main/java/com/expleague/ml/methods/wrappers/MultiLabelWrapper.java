package com.expleague.ml.methods.wrappers;

import com.expleague.commons.func.WeakListenerHolder;
import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.Trans;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.MultiLabelTools;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.multilabel.MultiLabelModel;
import com.expleague.ml.TargetFunc;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

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
    List<Consumer> internListeners = new ArrayList<>();
    if (strong instanceof WeakListenerHolder) {
      for (WeakReference<Consumer<? super Trans>> externalListenerRef : listeners) {
        final Consumer<? super Trans> externalListener = externalListenerRef.get();
        if (externalListener != null) {
          final Consumer<Trans> internListener = externalListener::accept;
          internListeners.add(internListener);
          ((WeakListenerHolder) strong).addListener(internListener);
        }
      }
    }

    final Trans model = strong.fit(learn, targetFunc);
    return MultiLabelTools.extractMultiLabelModel(model);
  }
}
