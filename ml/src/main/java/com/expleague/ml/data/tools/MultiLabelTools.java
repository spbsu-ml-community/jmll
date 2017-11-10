package com.expleague.ml.data.tools;

import com.expleague.ml.func.FuncEnsemble;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.math.Func;
import com.expleague.ml.cli.output.printers.MultiLabelLogitProgressPrinter;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.models.multilabel.MultiLabelBinarizedModel;
import com.expleague.ml.models.multilabel.MultiLabelModel;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * User: qdeee
 * Date: 20.03.15
 */
public final class MultiLabelTools {
  public static MultiLabelModel extractMultiLabelModel(final Function model) {
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

  public static void makeOVRReport(final Pool<?> learn, final Pool<?> test, final Function model, final int period) {
    if (model instanceof MultiLabelBinarizedModel) {
      final FuncJoin internModel = ((MultiLabelBinarizedModel) model).getInternModel();

      final MultiLabelLogitProgressPrinter progressPrinter = new MultiLabelLogitProgressPrinter(learn, test, period);
      final FuncEnsemble<?>[] perLabelModels = ArrayTools.map(internModel.dirs, FuncEnsemble.class, argument -> (FuncEnsemble<?>) argument);
      final int ensembleSize = perLabelModels[0].size();
      final int labelsCount = perLabelModels.length;

      final double step = perLabelModels[0].weights.get(0);
      final List<FuncJoin> weakModels = new ArrayList<>();
      for (int t = 0; t < ensembleSize; t++) {
        final Func[] functions = new Func[labelsCount];
        for (int c = 0; c < labelsCount; c++) {
          functions[c] = perLabelModels[c].models[t];
        }
        weakModels.add(new FuncJoin(functions));
        progressPrinter.accept(new Ensemble<>(weakModels, step));
      }
    } else {
      throw new UnsupportedOperationException("Can't extract intern FuncJoin model");
    }
  }
}
