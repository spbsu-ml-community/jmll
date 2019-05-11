package com.expleague.hist;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;
import com.expleague.ml.GridTools;
import com.expleague.ml.cli.output.printers.DefaultProgressPrinter;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.AUCLogit;
import com.expleague.ml.loss.L2Reg;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.PoolFeatureMeta;
import com.expleague.ml.meta.TargetMeta;
import com.expleague.ml.meta.items.QURLItem;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.ModelTools;
import com.expleague.ml.models.ObliviousTree;
import gnu.trove.list.array.TIntArrayList;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.exp;

public class HistogramMNISTExperiment {
  public static void main(String[] args) throws IOException, ClassNotFoundException {
    for (int num = 0; num < 10; num++) {
      System.out.println("Builing model for " + num);
      buildModelAndHist(num);
    }
  }

  private static void buildModelAndHist(int num) throws IOException, ClassNotFoundException {
    final Pool<QURLItem> mnistPoolTrain = DataTools.loadFromFeaturesTxt("/Users/solar/data/mnist/train.tsv");

    final Path modelPath = Paths.get("/Users/solar/data/mnist/" + num + ".model");
    final Ensemble<ObliviousTree> ovRModel;
    if (!Files.exists(modelPath)) {
      final Pool<QURLItem> mnistPoolTest = DataTools.loadFromFeaturesTxt("/Users/solar/data/mnist/valid.tsv");
      createNumTarget(mnistPoolTrain, num);
      createNumTarget(mnistPoolTest, num);
      final GradientBoosting<LLLogit> optimization = new GradientBoosting<>(
          new GreedyObliviousTree<>(GridTools.medianGrid(mnistPoolTrain.vecData(), 32), 6),
          L2Reg.class,
          2000,
          0.02
      );

      final DefaultProgressPrinter progressPrinter = new DefaultProgressPrinter(
          mnistPoolTrain,
          mnistPoolTest,
          mnistPoolTrain.target("" + num, LLLogit.class),
          new Func[]{mnistPoolTest.target("" + num, AUCLogit.class)},
          10
      );
      optimization.addListener(progressPrinter);

      //noinspection unchecked
      ovRModel = (Ensemble<ObliviousTree>) optimization.fit(mnistPoolTrain.vecData(), mnistPoolTrain.target("" + num, LLLogit.class));
      DataTools.writeModel(ovRModel, mnistPoolTrain.features(), modelPath);
    }
    else ovRModel = DataTools.readModel(Files.newInputStream(modelPath));

    final ModelTools.CompiledOTEnsemble compile = ModelTools.compile(ovRModel);
    histograms(mnistPoolTrain, DataTools.grid(ovRModel), compile.getEntries(), TIntArrayList.wrap(new int[0]), num);
  }

  private static void createNumTarget(Pool<QURLItem> mnistPool, int num) {
    final Vec classLabel = (Vec)mnistPool.target(0);
    final Vec numTarget = classLabel.stream().map(x -> x == num ? 1 : 0).collect(VecBuilder::new, VecBuilder::append, VecBuilder::addAll).build();
    mnistPool.addTarget(TargetMeta.create("" + num, "" + num, FeatureMeta.ValueType.VEC), numTarget);
  }

  private static void histograms(Pool<?> pool, BFGrid grid, List<ModelTools.CompiledOTEnsemble.Entry> entries, TIntArrayList histogramPath, int num) throws IOException {
    final VecDataSet vds = pool.vecData();
    final BinarizedDataSet bds = vds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    Vec weights = new ArrayVec(grid.rows());
    IntStream.range(0, grid.rows()).parallel().forEach(i -> {
      final BFGrid.Row row = grid.row(i);
      final PoolFeatureMeta meta = pool.features()[row.findex()];
      //      System.out.print(meta.id());
      double value = 0;
      double total = 0;
      final int[] path = histogramPath.toArray();
      for (int bin = 0; bin < row.size(); bin++) {
        final BFGrid.Feature binaryFeature = row.bf(bin);
        final List<ModelTools.CompiledOTEnsemble.Entry> vfEntries =
            entries.parallelStream()
                .filter(e -> ArrayTools.supset(e.getBfIndices(), path))
                .filter(e -> ArrayTools.indexOf(binaryFeature.index(), e.getBfIndices()) >= 0)
                .collect(Collectors.toList());
        final double[] weight = expectedWeight(grid, pool.vecData(), bds, vfEntries);
//        total += weight[0] / pool.size(); // original
        value += weight[0] / weight[1];
        total += Math.signum(value) * MathTools.sqr(value) * weight[1] / pool.size(); // c v^2
        //        if (Math.abs(weight) > MathTools.EPSILON)
        //          System.out.print(String.format("\t%d:%.3g:%.4g", bin, row.condition(bin), total));
      }
      weights.set(i, total);
//      System.out.println(total);
    });

    VecTools.scale(weights, 1./VecTools.maxMod(weights));

    final int scale = 10;
    BufferedImage jpgImage = new BufferedImage(28 * scale, 28 * scale, BufferedImage.TYPE_INT_RGB);

    Graphics2D g = jpgImage.createGraphics();
    {
      final float grade = (float) (1. / 2.);
      g.setBackground(new Color(grade, grade, grade));
      g.clearRect(0, 0, 28 * scale, 28 * scale);
    }

    for (int i = 0; i < grid.rows(); i++) {
      int x = i % 28 * scale;
      int y = i / 28 * scale;
      final float grade = (float) (1. / (1. + exp(-50 * weights.get(i))));
      g.setColor(new Color(grade, grade, grade));
      g.fillRect(x, y, scale, scale);
    }

    g.dispose();

    final ImageWriter pngWriter = ImageIO.getImageWritersByFormatName("png").next();
    pngWriter.setOutput(ImageIO.createImageOutputStream(Files.newOutputStream(Paths.get("/Users/solar/data/mnist/" + num + ".png"))));
    pngWriter.write(null, new IIOImage(jpgImage, null, null), null);
    pngWriter.dispose();
  }

  private static double[] expectedWeight(BFGrid grid, VecDataSet vds, BinarizedDataSet bds, List<ModelTools.CompiledOTEnsemble.Entry> vfEntries) {
    double total = 0;
    double count = 0;
    final int power = vds.length();
    for (int j = 0; j < power; j++) {
      final int finalJ = j;
      final double value = vfEntries.stream()
          .filter(entry -> {
            final int[] bfIndices = entry.getBfIndices();
            final int length = bfIndices.length;
            for (int i = 0; i < length; i++) {
              if (!grid.bf(bfIndices[i]).value(finalJ, bds))
                return false;
            }
            return true;
          })
          .mapToDouble(ModelTools.CompiledOTEnsemble.Entry::getValue)
          .sum();
      if (Math.abs(value) > 0) {
        total += value;
        count++;
      }
    }
    return new double[]{total, count};
  }
}
