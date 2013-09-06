package com.spbsu.ml;

import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.GreedyObliviousRegressionTree;
import com.spbsu.ml.models.AdditiveModel;
import gnu.trove.TByteArrayList;

import java.io.*;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

/**
 * User: solar
 * Date: 09.12.12
 * Time: 15:23
 */
public class HWScore {
  public static void main(String[] args) throws IOException {
    if (args.length >= 2 && "-dimred".equals(args[0])) {
      final DataSet learn = transform(args[1], new GZIPInputStream(HWScore.class.getClassLoader().getResourceAsStream("com/spbsu/ml/features.txt.gz")));
      final DataSet test = transform(args[1], new GZIPInputStream(HWScore.class.getClassLoader().getResourceAsStream("com/spbsu/ml/featuresTest.txt.gz")));
      final BFGrid grid = GridTools.medianGrid(learn, 32);
      final FastRandom rng = new FastRandom();
      final GradientBoosting boosting = new GradientBoosting(new GreedyObliviousRegressionTree(rng, learn, grid, 6), 2000, 0.005, rng);
      final ScoreCalcer score = new ScoreCalcer(test);
      boosting.addProgressHandler(score);
      boosting.fit(learn, new L2Loss(learn.target()));
      System.out.println("Best score: " + score.bestScore + " reached at iteration " + score.bestIter + ". Greed size: " + grid.size());
    }
    if (args.length >= 2 && "-fit".equals(args[0])) {
      final TByteArrayList result = StreamTools.transformByExternalCommand(args[1], new GZIPInputStream(HWScore.class.getClassLoader().getResourceAsStream("com/spbsu/ml/featuresTest.txt.gz")));
      final LineNumberReader lnr = new LineNumberReader(new InputStreamReader(new ByteArrayInputStream(result.toNativeArray())));
      final DataSet test = DataTools.loadFromFeaturesTxt(new InputStreamReader(new GZIPInputStream(HWScore.class.getClassLoader().getResourceAsStream("com/spbsu/ml/featuresTest.txt.gz")), "UTF-8"));

      String line;
      int index = 0;
      double loss = 0;
      while ((line = lnr.readLine()) != null) {
        final StringTokenizer tok = new StringTokenizer(line, "\t");
//        tok.nextToken();
        final double y = Double.parseDouble(tok.nextToken());
        double diff = test.target().get(index) - y;
        loss += diff * diff;
        index++;
      }

      System.out.println("Score: " + Math.sqrt(loss/test.power()));
    }
  }

  private static DataSet transform(String command, InputStream input) throws IOException {
    TByteArrayList bytes = StreamTools.transformByExternalCommand(command, input);
    return DataTools.loadFromFeaturesTxt(new StringReader(new String(bytes.toNativeArray())));
  }

  private static class ScoreCalcer implements ProgressHandler {
    final Vec current;
    private final DataSet ds;
    public double bestScore = Double.MAX_VALUE;
    public int bestIter;
    private int iteration = 0;

    public ScoreCalcer(DataSet ds) {
      this.ds = ds;
      current = new ArrayVec(ds.power());
    }

    @Override
    public void progress(Model partial) {
      iteration++;
      if (partial instanceof AdditiveModel) {
        final AdditiveModel additiveModel = (AdditiveModel) partial;
        final Model increment = (Model)additiveModel.models.get(additiveModel.models.size() - 1);
        final DSIterator iter = ds.iterator();
        int index = 0;
        while (iter.advance()) {
          current.adjust(index++, additiveModel.step * increment.value(iter.x()));
        }
      }
      else {
        final DSIterator iter = ds.iterator();
        int index = 0;
        while (iter.advance()) {
          current.set(index++, partial.value(iter.x()));
        }
      }
      double score = VecTools.distance(current, ds.target()) / Math.sqrt(ds.power());
      if (score <= bestScore) {
        bestScore = score;
        bestIter = iteration;
      }
    }
  }
}
