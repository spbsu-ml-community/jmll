package com.expleague.nn;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.func.generic.LogSoftMax;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;

import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.IntStream;

public class MNISTUtils {
  public static final int heightIn = 28;
  public static final int widthIn = 28;
  public static final int nClasses = 10;
  public static final int numTrainSamples = 60000;
  public static final int numTestSamples = 10000;
  private static final String projectDir = System.getProperty("user.dir");
  private static final String urlTrain = "https://pjreddie.com/media/files/mnist_train.csv";
  private static final String urlTest = "https://pjreddie.com/media/files/mnist_test.csv";
  private static final String pathTrain = projectDir + "/ml/src/main/resources/mnist_train.csv";
  private static final String pathTest = projectDir + "/ml/src/main/resources/mnist_test.csv";

  public static void downloadFile(String url, String path) {
    try {
      final Path targetTrain = Paths.get(path);
      if (Files.notExists(targetTrain)) {
        URL website = new URL(url);
        try (InputStream in = website.openStream()) {
          Files.copy(in, targetTrain, StandardCopyOption.REPLACE_EXISTING);
        }
        catch (IOException e) {
          e.printStackTrace();
        }
      }
    } catch (MalformedURLException e) {
      e.printStackTrace();
    }
  }

  public static void readMnist(final int[] trainLabels, final Mx trainSamples,
                               final int[] testLabels, final Mx testSamples) {
    try {
      Path fileTrain = Paths.get(pathTrain);
      Path fileTest = Paths.get(pathTest);

      if (Files.notExists(fileTrain)) {
        downloadFile(urlTrain, pathTrain);
        fileTrain = Paths.get(pathTrain);
      }

      if (Files.notExists(fileTest)) {
        downloadFile(urlTest, pathTest);
        fileTest = Paths.get(pathTest);
      }

      Files.lines(fileTrain)
          .forEach(new Consumer<String>() {
            int counter = 0;

            @Override
            public void accept(String s) {
              int[] data = Arrays.stream(s.split(","))
                  .mapToInt(Integer::parseInt).toArray();
              trainLabels[counter] = data[0];
              for (int i = 1; i < data.length; i++) {
                trainSamples.set(counter, i - 1, data[i]);
              }
              counter++;
            }
          });

      Files.lines(fileTest)
          .forEach(new Consumer<String>() {
            int counter = 0;

            @Override
            public void accept(String s) {
              int[] data = Arrays.stream(s.split(","))
                  .mapToInt(Integer::parseInt).toArray();
              testLabels[counter] = data[0];
              for (int i = 1; i < data.length; i++) {
                testSamples.set(counter, i - 1, data[i]);
              }
              counter++;
            }
          });
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static void testSoftmaxModel(Function<Vec, Vec> model,
                                      int[] trainLabels, Mx trainSamples,
                                      int[] testLabels, Mx testSamples) {
    int trainCounter = 0;
    for (int i = 0; i < trainLabels.length; i++) {
      Vec result = model.apply(trainSamples.row(i));
      final double sumExp = LogSoftMax.sumExp(result);
      double max = Double.NEGATIVE_INFINITY;
      int max_idx = -1;
      for (int j = 0; j < result.dim(); j++) {
        double value = Math.exp(result.get(j)) / sumExp;
        if (value > max) {
          max = value;
          max_idx = j;
        }
      }

      if (max_idx == trainLabels[i]) {
        trainCounter++;
      }
    }

    int testCounter = 0;
    for (int i = 0; i < testLabels.length; i++) {
      Vec result = model.apply(testSamples.row(i));
      final double sumExp = LogSoftMax.sumExp(result);
      double max = Double.NEGATIVE_INFINITY;
      int max_idx = -1;
      for (int j = 0; j < result.dim(); j++) {
        double value = Math.exp(result.get(j)) / sumExp;
        if (value > max) {
          max = value;
          max_idx = j;
        }
      }

      if (max_idx == testLabels[i]) {
        testCounter++;
      }
    }

    System.out.println("train accuracy " + ((double) trainCounter) / trainLabels.length);
    System.out.println("test accuracy " + ((double) testCounter) / testLabels.length);
  }

  public static void testMLLLogitModel(Function<Vec, Vec> model,
                                       IntSeq testLabels, Mx testSamples) {
    final Vec predict = new ArrayVec(testSamples.rows());
    final BlockwiseMLLLogit loss = new BlockwiseMLLLogit(testLabels, new VecDataSetImpl(testSamples, null));
    IntStream.range(0, testSamples.rows()).parallel().forEach(i -> {
      final Vec result = model.apply(testSamples.row(i));
      final Vec prob = loss.prob(result, new ArrayVec(loss.blockSize() + 1));
      predict.set(i, VecTools.argmax(prob));
    });
    final IntSeq labels = loss.labels();
    int acc = 0;
    for (int i = 0; i < predict.dim(); i++) {
      acc += predict.get(i) == labels.intAt(i) ? 1 : 0;
    }
    System.out.println("test accuracy " + ((double) acc) / predict.dim());
  }
}
