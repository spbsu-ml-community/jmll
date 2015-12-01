package com.spbsu.ml.models.multilabel;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Trans;

/**
 * User: qdeee
 * Date: 22.03.15
 */
public interface MultiLabelModel extends Trans {
  Vec predictLabels(Vec x);

  Mx predictLabelsAll(Mx mx);

  abstract class Stub extends Trans.Stub implements MultiLabelModel {
    @Override
    public Vec trans(final Vec x) {
      return predictLabels(x);
    }

    @Override
    public Mx predictLabelsAll(final Mx mx) {
      return transAll(mx);
    }
  }
}
