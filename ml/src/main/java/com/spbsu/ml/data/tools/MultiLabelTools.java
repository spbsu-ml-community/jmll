package com.spbsu.ml.data.tools;

import com.spbsu.ml.Trans;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.models.multilabel.MultiLabelBinarizedModel;
import com.spbsu.ml.models.multilabel.MultiLabelModel;

/**
 * User: qdeee
 * Date: 20.03.15
 */
public final class MultiLabelTools {
  public static MultiLabelModel extractMultiLabelModel(final Trans model) {
    MultiLabelModel multiLabelModel = null;
    if (model instanceof MultiLabelModel) {
      multiLabelModel = (MultiLabelModel) model;
    } else if (model instanceof Ensemble && ((Ensemble) model).last() instanceof FuncJoin) {
      final FuncJoin joined = MCTools.joinBoostingResult((Ensemble) model);
      multiLabelModel = new MultiLabelBinarizedModel(joined);
    }

    if (multiLabelModel == null) {
      throw new UnsupportedOperationException("Can't extract multi-label model");
    }

    return multiLabelModel;
  }
}
