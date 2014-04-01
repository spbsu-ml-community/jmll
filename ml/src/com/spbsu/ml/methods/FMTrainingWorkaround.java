package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.MxIterator;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.text.StringUtils;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.models.FMModel;

import java.io.*;

/**
 * User: qdeee
 * Date: 24.03.14
 */
public class FMTrainingWorkaround implements Optimization {

  private final static String LIBFM_PATH = System.getProperty("user.dir") + "/libfm";
  private String task;
  private String dim; // e.g, "1/1/8"
  private String iters;
  private String others;

  public FMTrainingWorkaround(String task, String dim, String iters, String others) {
    this.task = task;
    this.dim = dim.replace('/', ',');
    this.iters = iters;
    this.others = others;
  }

  public FMTrainingWorkaround(final String task, final String dim, final String iters) {
    this(task, dim, iters, "");
  }

  @Override
  public Trans fit(final DataSet learn, final Func func) {
    float minTarget = Float.MAX_VALUE;
    float maxTarget = Float.MIN_VALUE;
    for (int i = 0; i < learn.power(); i++) {
      final double t = learn.target().get(i);
      if (minTarget > t)
        minTarget = (float) t;
      if (maxTarget < t)
        maxTarget = (float) t;
    }
    int numFeatures = learn.xdim();
    int numRows = learn.power();
    long numValues = 0;
    MxIterator mxIterator = learn.data().nonZeroes();
    while (mxIterator.advance()) {
      numValues++;
    }

    try {
      final String[] params = {
          LIBFM_PATH,
          "-task", task,
          "-dim", dim,
          "-iter", iters,
          "-verbosity",
          others
      };
      final String cmd = StringUtils.concatWithDelimeter(" ", params);
      final Process exec = Runtime.getRuntime().exec(cmd);
      final LineNumberReader reader = new LineNumberReader(new InputStreamReader(exec.getInputStream()));
      final OutputStreamWriter writer = new OutputStreamWriter(exec.getOutputStream());

      readInput(reader, false);

      //sending dataset parameters
      writer.write(String.valueOf(minTarget));
      writer.write("\n");
      writer.write(String.valueOf(maxTarget));
      writer.write("\n");
      writer.write(String.valueOf(numFeatures));
      writer.write("\n");
      writer.write(String.valueOf(numRows));
      writer.write("\n");
      writer.write(String.valueOf(numValues));
      writer.write("\n");
      writer.flush();

      readInput(reader, false);

      //sending dataset
      for (DSIterator iter = learn.iterator(); iter.advance(); ) {
        String target = String.valueOf(iter.y());
        Vec row = VecTools.copySparse(iter.x());
        final String entry = String.format("%s %s\n", target, row.toString());
        writer.write(entry);
      }
      writer.flush();

      System.out.println("upload is finished");
      readInput(reader, true);

      //read result model
      StringBuilder modelStr = new StringBuilder();
      modelStr.append(reader.readLine());
      modelStr.append("\n");
      modelStr.append(reader.readLine());
      modelStr.append("\n");
      modelStr.append(reader.readLine());
      ModelsSerializationRepository serializationRepository = new ModelsSerializationRepository();
      final FMModel read = serializationRepository.read(modelStr, FMModel.class);
      return read;
    } catch (IOException e) {
      e.printStackTrace();
    }

    return null;
  }

  private void readInput(LineNumberReader reader, boolean blocking) throws IOException {
    String line;
    while ((line = reader.readLine()) != null && (reader.ready() || blocking) && (!line.equals("FM model"))) {
      System.out.println(line);
    }
  }
}
