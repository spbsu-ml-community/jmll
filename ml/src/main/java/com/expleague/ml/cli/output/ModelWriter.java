package com.expleague.ml.cli.output;

import com.expleague.commons.io.StreamTools;
import com.expleague.ml.BFGrid;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.dynamicGrid.interfaces.DynamicGrid;
import org.jetbrains.annotations.Nullable;

import java.io.File;
import java.io.IOException;
import java.util.function.Function;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class ModelWriter {
  private final String outName;

  public ModelWriter(final String outName) {
    this.outName = outName;
  }

  public void tryWriteBinFormula(final Function result) throws IOException {
    DataTools.writeBinModel(result, new File(outName + ".model"));
  }

  public void tryWriteGrid(final Function result) throws IOException {
    final @Nullable BFGrid grid = DataTools.grid(result);
    if (grid != null) {
      StreamTools.writeChars(DataTools.SERIALIZATION.write(grid), new File(outName + ".grid"));
    }
  }

  public void tryWriteDynamicGrid(final Function result) throws IOException {
    final @Nullable DynamicGrid dynamicGrid = DataTools.dynamicGrid(result);
    if (dynamicGrid != null) {
      StreamTools.writeChars(DataTools.SERIALIZATION.write(dynamicGrid), new File(outName + ".dgrid"));
    }
  }

  public void writeModel(final Function result) throws IOException {
    DataTools.writeModel(result, new File(outName + ".model"));
  }
}

