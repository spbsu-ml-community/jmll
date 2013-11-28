package com.spbsu.ml;

import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.func.Ensemble;
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
      final GradientBoosting<L2> boosting = new GradientBoosting<L2>(new GreedyObliviousTree<L2>(grid, 6), 2000, 0.005, rng);
      final ScoreCalcer score = new ScoreCalcer(test);
      boosting.addListener(score);
      boosting.fit(learn, new L2(learn.target()));
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
    private Vec current;
    private final DataSet ds;
    public double bestScore = Double.MAX_VALUE;
    public int bestIter;
    private int iteration = 0;

    public ScoreCalcer(DataSet ds) {
      this.ds = ds;
      current = new ArrayVec(ds.power());
    }

    @Override
    public void invoke(Trans partial) {
      iteration++;
      if (partial instanceof Ensemble) {
        final Ensemble ensemble = (Ensemble) partial;
        final Trans increment = ensemble.last();
        VecTools.append(current, VecTools.scale(increment.transAll(ds.data()), ensemble.wlast()));
      }
      else {
        current = partial.transAll(ds.data());
      }
      double score = VecTools.distance(current, ds.target()) / Math.sqrt(ds.power());
      if (score <= bestScore) {
        bestScore = score;
        bestIter = iteration;
      }
    }
  }
}
