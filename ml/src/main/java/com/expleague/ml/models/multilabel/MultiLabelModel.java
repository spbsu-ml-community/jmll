package com.expleague.ml.models.multilabel;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;

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
