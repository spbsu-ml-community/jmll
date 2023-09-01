package com.expleague.pnfa;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.methods.seq.BootstrapSeqOptimization;
import com.expleague.ml.methods.seq.GradientSeqBoosting;
import com.expleague.ml.methods.seq.IntAlphabet;
import com.expleague.ml.methods.seq.PNFARegressor;
import com.expleague.ml.methods.seq.param.BettaMxParametrization;
import com.expleague.ml.methods.seq.param.BettaParametrization;
import com.expleague.ml.methods.seq.param.WeightParametrization;
import com.expleague.ml.methods.seq.param.WeightSquareParametrization;
import com.expleague.ml.optimization.impl.AdamDescent;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.Function;

public class RunMnist {
  private static final int WEIGHT_EPOCH_COUNT = 15;
  private static final int BATCH_SIZE = 6;
  private static final double WEIGHT_STEP = 0.0002;

  private static Options options = new Options();
  private static List<IntSeq> trainData, testData;
  private static IntSeq trainLabels, testLabels;
  private static Random random = new Random(239);

  static {
    options.addOption("stateCount", true, "stateCount");
    options.addOption("alpha", true, "alpha");
    options.addOption("addToDiag", true, "addToDiag");
    options.addOption("boostStep", true, "boostStep");
    options.addOption("dataset", true, "datasetPath");
    options.addOption("checkpointFolder", true, "checkpointFolder");
  }


  private static int readInt(byte[] bytes, int start, int len) {
    ByteBuffer wrapped = ByteBuffer.wrap(bytes, start, len); // big-endian by default
    return wrapped.getInt();
  }

  private static List<IntSeq> parseImages(byte[] bytes) {
    List<IntSeq> data = new ArrayList<>();

    int imageCount = readInt(bytes, 4, 4);
    if (imageCount * 28 * 28 + 16 != bytes.length) {
      throw new IllegalArgumentException("Expected " + (imageCount * 28 * 28 + 16) + " bytes, but found " + bytes.length);
    }

    for (int i = 0; i < imageCount; i++) {
      int[][] image = new int[28][29];
      int idx = 28 * 28 * i + 16;
      for (int j = 0; j < 28; j++) {
        for (int k = 0; k < 28; k++) {
          image[j][k] = bytes[idx] < 0 ? 1 : 0;
//          if (i == 2) {
//            System.out.print(image[j][k]);
//            if (k == 27) {
//              System.out.println();
//            }
//          }
          idx++;
        }
      }

      int[] seq = new int[28];
      for (int j = 0; j < 28; j++) {
        int col = 0;
        for (int k = 0; k < 14; k++) {
          col *= 2;
          if (image[2 * k][j] == 1 || image[2 * k + 1][j] == 1) {
            col += 1;
          }
        }
        seq[j] = col;
      }
      data.add(new IntSeq(seq));
    }

    return data;
  }

  private static IntSeq parseLabels(byte[] bytes) {
    int imageCount = readInt(bytes, 4, 4);
    if (imageCount + 8 != bytes.length) {
      throw new IllegalArgumentException("Expected " + (imageCount + 8) + " bytes, but found " + bytes.length);
    }

    int[] labels = new int[imageCount];
    for (int i = 0; i < imageCount; i++) {
      labels[i] = bytes[i + 8];
    }

    return new IntSeq(labels);
  }

  private static void loadData(final String path) {
    try {
      trainData = parseImages(Files.readAllBytes(Paths.get(path, "train-images.idx3-ubyte")));
      trainLabels = parseLabels(Files.readAllBytes(Paths.get(path, "train-labels.idx1-ubyte")));

      testData = parseImages(Files.readAllBytes(Paths.get(path, "t10k-images.idx3-ubyte")));
      testLabels = parseLabels(Files.readAllBytes(Paths.get(path, "t10k-labels.idx1-ubyte")));
    }
    catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) throws IOException, ParseException {
    final CommandLineParser parser = new GnuParser();
    final CommandLine command = parser.parse(options, args);

    loadData(command.getOptionValue("dataset"));

    final double addToDiag = Double.parseDouble(command.getOptionValue("addToDiag"));

    BettaParametrization bettaParametrization = new BettaMxParametrization(addToDiag);
    WeightParametrization weightParametrization= new WeightSquareParametrization(bettaParametrization);

    PNFARegressor<Integer, WeightedL2> pnfa = new PNFARegressor<>(
        Integer.parseInt(command.getOptionValue("stateCount")),
        Integer.parseInt(command.getOptionValue("stateCount")),
        10,
        new IntAlphabet(1 << 14),
        Double.parseDouble(command.getOptionValue("alpha")),
        0.001,
        addToDiag,
        0.0,
        random,
        new AdamDescent(random, 1, BATCH_SIZE, WEIGHT_STEP),
        bettaParametrization,
        weightParametrization
    );

    Vec trainTarget = new ArrayVec(trainData.size() * 10);
    for (int i = 0; i < trainLabels.length(); i++) {
      trainTarget.set(i * 10 + trainLabels.at(i), 1);
    }
    DataSet<Seq<Integer>> dataSet = new DataSet.Stub<Seq<Integer>>(null) {
      @Override
      public Seq<Integer> at(int i) {
        return trainData.get(i);
      }

      @Override
      public int length() {
        return trainData.size();
      }

      @Override
      public Class<Seq<Integer>> elementType() {
        return null;
      }
    };

    GradientSeqBoosting<Integer, WeightedL2> boosting =
        new GradientSeqBoosting<>(new BootstrapSeqOptimization<>(pnfa, new FastRandom(123), 1), 30, 0.1);
    Consumer<Function<Seq<Integer>,Vec>> listener = model -> {
      System.out.println("Train accuracy: " + getAccuracy(model, trainData, trainLabels));
      System.out.println("Test accuracy: " + getAccuracy(model, testData, testLabels));
    };
    boosting.addListener(listener);

    Function<Seq<Integer>, Vec> model = boosting.fit(dataSet, new WeightedL2(trainTarget, dataSet));

    System.out.println("Train accuracy: " + getAccuracy(model, trainData, trainLabels));
    System.out.println("Test accuracy: " + getAccuracy(model, testData, testLabels));
    //
//    PNFABasecall basecall = new PNFABasecall(
//        Paths.get(command.getOptionValue("dataset")),
//        Paths.get(command.getOptionValue("checkpointFolder")),
//        Integer.parseInt(command.getOptionValue("stateCount")),
//        Integer.parseInt(command.getOptionValue("alphaShrink")),
//        Double.parseDouble(command.getOptionValue("lambda")),
//        Double.parseDouble(command.getOptionValue("addToDiag")),
//        Double.parseDouble(command.getOptionValue("boostStep")),
//        Double.parseDouble(command.getOptionValue("trainPart")),
//        Double.parseDouble(command.getOptionValue("testPart")),
//        239,
//        false
//    );
  }

  private static double getAccuracy(Function<Seq<Integer>, Vec> model, List<IntSeq> data, IntSeq labels) {
    double correct = 0;
    for (int i = 0; i < data.size(); i++) {
      if (VecTools.argmax(model.apply(data.get(i))) == labels.at(i)) {
        correct++;
      }
    }

    return correct / data.size();
  }
}
