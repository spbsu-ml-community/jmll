package com.spbsu.exp.cart;

import com.spbsu.commons.func.Processor;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;

import java.io.*;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;

public class DataLoader {
  static public DataFrame bootstrap(DataFrame data, long seed) {
    FastRandom rnd = new FastRandom(seed);

    VecBuilder targetBuilder = new VecBuilder();
    VecBuilder featureBuilder = new VecBuilder();
    int countFeatures = data.getLearnFeatures().xdim();
    for (int i = 0; i < data.getLearnFeatures().length(); i++) {
      int cnt_i = rnd.nextPoisson(1.);
      Vec curVec = data.getLearnFeatures().at(i);
      for (int j = 0; j < cnt_i; j++) {
        for (int k = 0; k < curVec.dim(); k++) {
          featureBuilder.add(curVec.get(k));
        }
        targetBuilder.append(data.getLearnTarget().get(i));
      }
    }

    Mx dataMX = new VecBasedMx(countFeatures,
            featureBuilder.build());
    VecDataSet learnFeatures = new VecDataSetImpl(dataMX, null);

    return new DataFrame(learnFeatures, targetBuilder.build(),
            data.getTestFeatures(), data.getTestTarget());
  }

  static public class DataFrame {
    final private VecDataSet learnFeatures;
    final private Vec learnTarget;
    final private VecDataSet testFeatures;
    final private Vec testTarget;

    DataFrame(VecDataSet learnFeatures, Vec learnTarget, VecDataSet testFeatures, Vec testTarget) {
      this.learnFeatures = learnFeatures;
      this.learnTarget = learnTarget;
      this.testFeatures = testFeatures;
      this.testTarget = testTarget;
    }

    VecDataSet getLearnFeatures() {
      return learnFeatures;
    }

    Vec getLearnTarget() {
      return learnTarget;
    }

    VecDataSet getTestFeatures() {
      return testFeatures;
    }

    Vec getTestTarget() {
      return testTarget;
    }

    private void toFile(VecDataSet dataSet, File file) throws FileNotFoundException {
      PrintWriter writer = new PrintWriter(file);
      Mx trainData = dataSet.data();
      for (int i = 0; i < dataSet.length(); i++) {
        for (int j = 0; j < dataSet.xdim(); j++) {
          String append = ", ";
          if (j == dataSet.xdim() - 1) {
            append = "\n";
          }
          writer.write(Double.toString(trainData.get(i, j)) + append);
        }
      }
      writer.close();
    }

    private void toFile(Vec vec, File file) throws FileNotFoundException {
      PrintWriter writer = new PrintWriter(file);
      for (int i = 0; i < vec.dim(); i++) {
        String append = "\n";
        writer.write(Double.toString(vec.at(i)) + append);
      }
      writer.close();
    }

    void toFile(String folder) throws FileNotFoundException {
      File trainFile = new File(folder + "train.csv");
      File testFile = new File(folder + "test.csv");
      File trainTargetFile = new File(folder + "train_target.csv");
      File testTargetFile = new File(folder + "test_target.csv");

      toFile(learnFeatures, trainFile);
      toFile(testFeatures, testFile);
      toFile(learnTarget, trainTargetFile);
      toFile(testTarget, testTargetFile);
    }
  }

  private static String getFullPath(String file, String directory) {
    return Paths.get(directory, file).toString();
  }

  private static Reader getReader(String fileName, String directory) throws IOException {
    return fileName.endsWith(".gz") ?
            new InputStreamReader(new GZIPInputStream(new FileInputStream(getFullPath(fileName, directory)))) :
            new FileReader(getFullPath(fileName, directory));
  }

  public static abstract class TestProcessor implements Processor<CharSequence> {
    public abstract VecBuilder getTargetBuilder();

    public abstract VecBuilder getFeaturesBuilder();

    public abstract int getFeaturesCount();

    public void wipe() {
      getFeaturesBuilder().clear();
      getTargetBuilder().clear();
    }
  }

  public static class KSHouseReadProcessor extends TestProcessor {
    private VecBuilder targetBuilder = new VecBuilder();
    private VecBuilder featuresBuilder = new VecBuilder();
    private static final int featureCount = 19;

    @Override
    public void process(CharSequence arg) {
      final CharSequence[] parts = CharSeqTools.split(arg, ',');
      targetBuilder.append(CharSeqTools.parseDouble(parts[2]));
      for (int i = 3; i <= 20; i++) {
        featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
      }
      featuresBuilder.append(CharSeqTools.parseDouble(parts[1]));
    }

    @Override
    public VecBuilder getTargetBuilder() {
      return targetBuilder;
    }

    @Override
    public VecBuilder getFeaturesBuilder() {
      return featuresBuilder;
    }

    @Override
    public int getFeaturesCount() {
      return featureCount;
    }
  }

  public static class CTSliceTestProcessor extends TestProcessor {
    private VecBuilder targetBuilder = new VecBuilder();
    private VecBuilder featuresBuilder = new VecBuilder();
    private int featureCount = 384;

    @Override
    public void process(CharSequence arg) {
      final CharSequence[] parts = CharSeqTools.split(arg, ',');
      targetBuilder.append(CharSeqTools.parseDouble(parts[385]));
      for (int i = 1; i < 385; i++) {
        featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
      }
    }

    @Override
    public VecBuilder getTargetBuilder() {
      return targetBuilder;
    }

    @Override
    public VecBuilder getFeaturesBuilder() {
      return featuresBuilder;
    }

    @Override
    public int getFeaturesCount() {
      return featureCount;
    }
  }

  public static class CancerTestProcessor extends TestProcessor {
    private VecBuilder targetBuilder = new VecBuilder();
    private VecBuilder featuresBuilder = new VecBuilder();
    private int featureCount = -1;

    @Override
    public void process(CharSequence arg) {
      final CharSequence[] parts = CharSeqTools.split(arg, ',');
      featureCount = parts.length - 2;
      if (parts[1].charAt(0) == 'M') {
        targetBuilder.append(-1);
      } else {
        targetBuilder.append(1);
      }
      for (int i = 2; i < featureCount + 2; i++) {
        featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
      }
    }

    @Override
    public VecBuilder getTargetBuilder() {
      return targetBuilder;
    }

    @Override
    public VecBuilder getFeaturesBuilder() {
      return featuresBuilder;
    }

    @Override
    public int getFeaturesCount() {
      return featureCount;
    }
  }

  public static class BaseDataReadProcessor extends TestProcessor {
    private VecBuilder targetBuilder = new VecBuilder();
    private VecBuilder featuresBuilder = new VecBuilder();
    private int featureCount = -1;

    @Override
    public void process(CharSequence arg) {
      final CharSequence[] parts = CharSeqTools.split(arg, '\t');
      targetBuilder.append(CharSeqTools.parseDouble(parts[1]));
      if (featureCount < 0)
        featureCount = parts.length - 4;
      else if (featureCount != parts.length - 4)
        throw new RuntimeException("\"Failed to parse line \" + index + \":\"");
      for (int i = 4; i < parts.length; i++) {
        featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
      }
    }

    @Override
    public VecBuilder getTargetBuilder() {
      return targetBuilder;
    }

    @Override
    public VecBuilder getFeaturesBuilder() {
      return featuresBuilder;
    }

    @Override
    public int getFeaturesCount() {
      return featureCount;
    }

    @Override
    public void wipe() {
      super.wipe();
      featureCount = -1;
    }
  }

  public static class HIGGSReadProcessor extends TestProcessor {
    private VecBuilder targetBuilder = new VecBuilder();
    private VecBuilder featuresBuilder = new VecBuilder();
    private int featureCount = 28;

    @Override
    public void process(CharSequence arg) {
      final CharSequence[] parts = CharSeqTools.split(arg, ',');
      int curAns = (int)CharSeqTools.parseDouble(parts[0]);
      if (curAns == 0) curAns = -1;
      targetBuilder.append(curAns);
      for (int i = 1; i <= 28; i++) {
        featuresBuilder.append(CharSeqTools.parseDouble(parts[i]));
      }

    }

    @Override
    public VecBuilder getTargetBuilder() {
      return targetBuilder;
    }

    @Override
    public VecBuilder getFeaturesBuilder() {
      return featuresBuilder;
    }

    @Override
    public int getFeaturesCount() {
      return featureCount;
    }
  }

  public static DataFrame readData(TestProcessor processor, String directory, String learnFileName, String testFileName)
          throws IOException {

    final Reader inLearn = getReader(learnFileName, directory);
    final Reader inTest = getReader(testFileName, directory);

    CharSeqTools.processLines(inLearn, processor);
    Mx data = new VecBasedMx(processor.getFeaturesCount(),
            processor.getFeaturesBuilder().build());
    VecDataSet learnFeatures = new VecDataSetImpl(data, null);
    Vec learnTarget = processor.getTargetBuilder().build();
    processor.wipe();
    CharSeqTools.processLines(inTest, processor);
    data = new VecBasedMx(processor.getFeaturesCount(),
            processor.getFeaturesBuilder().build());
    VecDataSet testFeatures = new VecDataSetImpl(data, null);
    Vec testTarget = processor.getTargetBuilder().build();
    return new DataFrame(learnFeatures, learnTarget, testFeatures, testTarget);
  }
}
