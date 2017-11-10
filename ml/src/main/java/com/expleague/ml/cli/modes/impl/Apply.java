package com.expleague.ml.cli.modes.impl;

import com.expleague.commons.io.StreamTools;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.MathTools;
import com.expleague.ml.BFGrid;
import com.expleague.ml.cli.JMLLCLI;
import com.expleague.ml.cli.builders.data.impl.DataBuilderClassic;
import com.expleague.ml.cli.builders.methods.grid.GridBuilder;
import com.expleague.ml.cli.modes.AbstractMode;
import com.expleague.ml.cli.modes.CliPoolReaderHelper;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.io.ModelsSerializationRepository;
import com.expleague.commons.seq.CharSeqBuilder;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.meta.DSItem;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.function.Function;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class Apply extends AbstractMode {

  public void run(final CommandLine command) throws MissingArgumentException, IOException, ClassNotFoundException {
    if (!command.hasOption(JMLLCLI.LEARN_OPTION) || !command.hasOption(JMLLCLI.MODEL_OPTION)) {
      throw new MissingArgumentException("Please, provide 'LEARN_OPTION' and 'MODEL_OPTION'");
    }

    final DataBuilderClassic dataBuilder = new DataBuilderClassic();
    dataBuilder.setLearnPath(command.getOptionValue(JMLLCLI.LEARN_OPTION));
    CliPoolReaderHelper.setPoolReader(command, dataBuilder);
    final Pool<? extends DSItem> pool = dataBuilder.create().getFirst();
    final VecDataSet vecDataSet = pool.vecData();

    final ModelsSerializationRepository serializationRepository;
    if (command.hasOption(JMLLCLI.GRID_OPTION)) {
      final GridBuilder gridBuilder = new GridBuilder();
      gridBuilder.setGrid(BFGrid.CONVERTER.convertFrom(StreamTools.readFile(new File(command.getOptionValue(JMLLCLI.GRID_OPTION)))));
      serializationRepository = new ModelsSerializationRepository(gridBuilder.create());
    } else {
      serializationRepository = new ModelsSerializationRepository();
    }

    try (final OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(getOutputName(command) + ".values"))) {
      final Function model = DataTools.readModel(command.getOptionValue(JMLLCLI.MODEL_OPTION, "features.txt.model"), serializationRepository);
      final CharSeqBuilder value = new CharSeqBuilder();

      for (int i = 0; i < pool.size(); i++) {
        value.clear();
        value.append(pool.data().at(i).id());
        value.append('\t');
//        value.append(MathTools.CONVERSION.convert(vecDataSet.parent().at(i), CharSequence.class));
//        value.append('\t');
//        value.append(MathTools.CONVERSION.convert(vecDataSet.at(i), CharSequence.class));
//        value.append('\t');
        if (model instanceof Func)
          value.append(MathTools.CONVERSION.convert(((Func) model).value(vecDataSet.at(i)), CharSequence.class));
        else if (model instanceof Ensemble && Func.class.isAssignableFrom(((Ensemble) model).componentType()))
          value.append(MathTools.CONVERSION.convert(((Ensemble) model).apply(vecDataSet.at(i)).get(0), CharSequence.class));
        else
          value.append(MathTools.CONVERSION.convert(model.apply(vecDataSet.at(i)), CharSequence.class));
        writer.append(value).append('\n');
      }
    }
  }
}
