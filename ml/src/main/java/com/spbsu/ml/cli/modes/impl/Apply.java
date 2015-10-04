package com.spbsu.ml.cli.modes.impl;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.seq.CharSeqBuilder;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.cli.builders.data.impl.DataBuilderClassic;
import com.spbsu.ml.cli.builders.methods.grid.GridBuilder;
import com.spbsu.ml.cli.modes.AbstractMode;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.meta.DSItem;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

import static com.spbsu.ml.cli.JMLLCLI.*;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class Apply extends AbstractMode {

  public void run(final CommandLine command) throws MissingArgumentException, IOException, ClassNotFoundException {
    if (!command.hasOption(LEARN_OPTION) || !command.hasOption(MODEL_OPTION)) {
      throw new MissingArgumentException("Please, provide 'LEARN_OPTION' and 'MODEL_OPTION'");
    }

    final DataBuilderClassic dataBuilder = new DataBuilderClassic();
    dataBuilder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    dataBuilder.setJsonFormat(command.hasOption(JSON_FORMAT));
    final Pool<? extends DSItem> pool = dataBuilder.create().getFirst();
    final VecDataSet vecDataSet = pool.vecData();

    final ModelsSerializationRepository serializationRepository;
    if (command.hasOption(GRID_OPTION)) {
      final GridBuilder gridBuilder = new GridBuilder();
      gridBuilder.setGrid(BFGrid.CONVERTER.convertFrom(StreamTools.readFile(new File(command.getOptionValue(GRID_OPTION)))));
      serializationRepository = new ModelsSerializationRepository(gridBuilder.create());
    } else {
      serializationRepository = new ModelsSerializationRepository();
    }

    try (final OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(getOutputName(command) + ".values"))) {
      final Computable model = DataTools.readModel(command.getOptionValue(MODEL_OPTION, "features.txt.model"), serializationRepository);
      final CharSeqBuilder value = new CharSeqBuilder();

      for (int i = 0; i < pool.size(); i++) {
        value.clear();
        value.append(pool.data().at(i).id());
        value.append('\t');
//        value.append(MathTools.CONVERSION.convert(vecDataSet.parent().at(i), CharSequence.class));
//        value.append('\t');
        value.append(MathTools.CONVERSION.convert(vecDataSet.at(i), CharSequence.class));
        value.append('\t');
        value.append(MathTools.CONVERSION.convert(model.compute(vecDataSet.at(i)), CharSequence.class));
        writer.append(value).append('\n');
      }
    }
  }
}
