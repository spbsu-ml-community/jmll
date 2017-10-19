package com.expleague.mcg;


import com.expleague.commons.io.StreamTools;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.data.tools.DataTools;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.exp;
import static java.lang.Math.sqrt;
import static org.bytedeco.javacpp.opencv_core.CV_8UC3;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;


/**
 * Experts League
 * Created by solar on 14.08.17.
 */
public class MSGMain {
  public static void main(String[] args) throws IOException {
    final opencv_core.Mat imageSrc = imread("image-in-2.jpeg");
    final opencv_core.Mat image = new opencv_core.Mat(32, 32, CV_8UC3);
    opencv_imgproc.resize(imageSrc, image, new opencv_core.Size(32, 32  ));
//    final opencv_core.Mat image = imread("input.jpg");
//    for (int i = 0; i < image.rows(); i++) {
//      for (int j = 0; j < image.cols(); j++) {
//        int val = i >= j ? 0 : 255;
//        image.row(i).col(j).put(opencv_core.Scalar.all(val));
//      }
//    }

    int size = image.rows() * image.cols();
    final Mx affinity = matrix(size);
    for (int i = 0; i < image.rows(); i++) {
      for (int j = 0; j < image.cols(); j++) {
        final int finalI = i;
        final int finalJ = j;
        final double sum = neighborhoodConvolve(i, j, image.cols(), image.rows(), 5, (y, x, distance) -> {
          double sim = sim(image.row(finalI).col(finalJ), image.row(y).col(x)) * Math.exp(-distance*distance/2/2);
          affinity.adjust(finalI * image.cols() + finalJ, y * image.cols() + x, -sim);
          return sim;
        });
        affinity.adjust(i * image.cols() + j, i * image.cols() + j, sum);
      }
    }

    StreamTools.writeChars(DataTools.SERIALIZATION.write(affinity), new File("affinity-mx.txt"));
    Mx q = matrix(affinity.columns());
    Mx sigma = matrix(affinity.columns());
    MxTools.eigenDecomposition(affinity, sigma, q);
    double[] eigenvalues = new double[affinity.columns()];

    for (int z = 0; z < affinity.columns(); z++) {
      eigenvalues[z] = sigma.get(z, z);
    }
    int[] eigOrder = ArrayTools.sequence(0, eigenvalues.length);
    ArrayTools.parallelSort(eigenvalues, eigOrder);

    final Mx depths = new VecBasedMx(image.cols(), new ArrayVec(image.cols() * image.rows()));
    double sum = 0;
    double sum2 = 0;
    for (int i = 0; i < image.rows(); i++) {
      for (int j = 0; j < image.cols(); j++) {
        double value = 0;

        for (int z = 0; z < affinity.columns(); z++) {
          int index = eigOrder[z];
          double weight = 1 / (0.01 + Math.sqrt(sigma.get(index, index)));
          value += weight * q.get(i * image.cols() + j, index);
        }
        sum += value;
        sum2 += value * value;
        depths.set(i, j, value);
      }
    }
    final double var = Math.sqrt((sum2 - sum * sum / size)/size);

    int[] coords = ArrayTools.sequence(0, depths.dim());
    ArrayTools.parallelSort(depths.toArray(), coords);

    int clusters[] = new int[depths.dim()];
    final TIntArrayList pool = new TIntArrayList();
    final TDoubleArrayList clusterDepths = new TDoubleArrayList();
    final List<PixelGenerationModel> clusterPGMS = new ArrayList<>();
    clusterDepths.add(-1);
    clusterPGMS.add(new PixelGenerationModel());
    int clustersCount = 1;
    for (int t = 0; t < coords.length; t++) {
      final int i = coords[t] / depths.columns();
      final int j = coords[t] % depths.columns();
      final double depthAtPoint = depths.get(i, j);
      pool.clear();
      neighborhoodConvolve(i, j, depths.columns(), depths.rows(), 1, (y, x, d) ->{
        pool.add(clusters[y * depths.columns() + x]);
        return 0;
      });
      if (pool.max() == -1) {
        clusters[i * depths.columns() + j] = -1; // border
        continue;
      }
      TIntArrayList values = new TIntArrayList(ArrayTools.values(pool.toArray()));
      //noinspection StatementWithEmptyBody
      values.remove(-1);
      //noinspection StatementWithEmptyBody
      values.remove(0);
      if (values.isEmpty()) { // create new cluster
        clusters[i * depths.columns() + j] = clustersCount++;
        clusterDepths.add(depthAtPoint);
        PixelGenerationModel pgm = new PixelGenerationModel();
        clusterPGMS.add(pgm);
        pgm.accept(image.row(i).col(j));
      }
      else {
        if (values.min() == values.max()) { // assign to nearest
          clusters[i * depths.columns() + j] = values.min();
          clusterPGMS.get(values.min()).accept(image.row(i).col(j));
        }
        else { // join clusters
          int deepCount = 1;
          int deepCluster = values.min();
          final PixelGenerationModel deepestModel = clusterPGMS.get(deepCluster);
          for (int v = 0; v < values.size(); v++) {
            final int currentCluster = values.get(v);
            if (deepCluster == currentCluster)
              continue;
            if (depthAtPoint - clusterDepths.get(currentCluster) > MathTools.EPSILON) {
              final PixelGenerationModel otherPgm = clusterPGMS.get(currentCluster);
              if (otherPgm.count > 3 && deepestModel.similarity(otherPgm) < 0.5) {
                deepCount++;
              }
            }
          }
          if (deepCount > 1) {
            clusters[i * depths.columns() + j] = -1; // border
          }
          else { // flood regions
            clusters[i * depths.columns() + j] = deepCluster;
            deepestModel.accept(image.row(i).col(j));
            for (int v = 0; v < values.size(); v++) {
              final int currentCluster = values.get(v);
              if (currentCluster == deepCluster)
                continue;
              for (int k = 0; k < clusters.length; k++) {
                if (currentCluster == clusters[k]) {
                  clusters[k] = deepCluster;
                  deepestModel.accept(image.row(k/image.cols()).col(k%image.cols()));
                }
              }
            }
          }
        }
      }
    }

//    for (int currentCluster : ArrayTools.values(clusters)) {
//      final TIntArrayList region = new TIntArrayList(clusters.length);
//      for (int k = 0; k < clusters.length; k++) {
//        if (currentCluster == clusters[k]) {
//          region.add(k);
//        }
//      }
//      region.trimToSize();
//      regions.put(currentCluster, region);
//    }
    for (int i = 0; i < image.rows(); i++) {
      for (int j = 0; j < image.cols(); j++) {
        System.out.print(" " + clusters[i * image.cols() + j]);
      }
      System.out.println();
    }
    outer:
    while (true) {
      VecTools.fill(depths, 0);
      for (int t = 0; t < clusters.length; t++) {
        if (clusters[t] > 0)
          continue;

        final int i = t / depths.columns();
        final int j = t % depths.columns();
        pool.clear();
        pool.add(-1);
        neighborhoodConvolve(i, j, depths.columns(), depths.rows(), Math.sqrt(2), (y, x, d) -> {
          pool.add(clusters[y * depths.columns() + x]);
          return 0;
        });

        final TIntArrayList values = new TIntArrayList(ArrayTools.values(pool.toArray()));
        values.remove(-1);
        if (values.size() == 1)
          continue;
        if (values.size() == 2) {
          int deepest = values.min();
          final PixelGenerationModel pgm1 = clusterPGMS.get(deepest);
          int other = values.max();
          final PixelGenerationModel pgm2 = clusterPGMS.get(other);
          if (pgm1.similarity(pgm2) > 0.05) { //merge
            for (int k = 0; k < clusters.length; k++) {
              if (other == clusters[k]) {
                clusters[k] = deepest;
                pgm1.accept(image.row(k/image.cols()).col(k%image.cols()));
              }
            }
            continue outer;
          }
        }
        depths.set(t, 1);
      }
      break;
    }
//    for (int c = 0; c < clustersCount; c++) {
//      final double increment = 1;
//      final TIntArrayList region = regions.get(c);
//      if (region == null)
//        continue;
//
//      region.forEach(index -> {
//        depths.adjust(index, increment);
//        return true;
//      });
//    }
//
    double max = VecTools.max(depths);
    for (int i = 0; i < image.rows(); i++) {
      opencv_core.Mat row = image.row(i);
      for (int j = 0; j < image.cols(); j++) {
        if (depths.get(i, j) > 0)
          row.col(j).put(opencv_core.Scalar.all(depths.get(i, j)/max * 255));
      }
    }

    imwrite("image-haha.jpeg", image);
  }

  @NotNull
  private static VecBasedMx matrix(int size) {
//    return new VecBasedMx(size, new SparseVec(MathTools.sqr(size)));
    return new VecBasedMx(size, new ArrayVec(MathTools.sqr(size)));
  }

//  private static double sim(opencv_core.Mat col, opencv_core.Mat col1) {
//    double module1 = 0;
//    double module2 = 0;
//    double product = 0;
//    ByteBuffer buffer = col.createBuffer();
//    ByteBuffer buffer1 = col1.createBuffer();
//    for (int i = 0; i < 3; i++) {
//      double a = buffer.get();
//      double b = buffer1.get();
//      if (b < 0)
//        b = 256 + b;
//      if (a < 0)
//        a = 256 + a;
//      module1 += a * a;
//      module2 += b * b;
//      product += a * b;
//    }
//    if (module1 == 0 || module2 == 0)
//      return 0;
//    return product / Math.sqrt(module1 * module2);
//  }

//  private static double sim(opencv_core.Mat col, opencv_core.Mat col1) {
//    double value = 0;
//    {
//      ByteBuffer buffer = col.createBuffer();
//      ByteBuffer buffer1 = col1.createBuffer();
//      for (int i = 0; i < 3; i++) {
//        double a = buffer.get();
//        double b = buffer1.get();
//        if (b < 0)
//          b = 256 + b;
//        if (a < 0)
//          a = 256 + a;
//        value += MathTools.sqr(a /255 - b / 255);
//      }
//    }
//    return exp(-value);
//  }

  private static final double labdelta = 6. / 29.;
  private static double labf(double x) {
    if (x > labdelta * labdelta * labdelta)
      return Math.pow(x, 1. / 3.);
    return x / 3. / labdelta / labdelta + 4. / 29.;
  }
  private static double sim(opencv_core.Mat col, opencv_core.Mat col1) {
    double value = 0;
    {
      ByteBuffer buffer = col.createBuffer();
      ByteBuffer buffer1 = col1.createBuffer();
      double r1 = norm(buffer.get()), r2 = norm(buffer1.get());
      double g1 = norm(buffer.get()), g2 = norm(buffer1.get());
      double b1 = norm(buffer.get()), b2 = norm(buffer1.get());
      double L1 = 116 * labf(g1) - 16, L2 = 116 * labf(g2) - 16;
      double A1 = 500 * (labf(r1 / 0.95047) - labf(g1)), A2 = 500 * (labf(r2 / 0.95047) - labf(g2));
      double B1 = 200 * (labf(g1) - labf(b1/1.08883)), B2 = 200 * (labf(g2) - labf(b2/1.08883));
      value = ((L1 - L2) * (L1 - L2) + (A1 - A2) * (A1 - A2) + (B1 - B2) * (B1 - B2));
//      value = ((A1 - A2) * (A1 - A2) + (B1 - B2) * (B1 - B2));
    }
//    System.out.println(value + " " + Math.exp(-value/10));
    return Math.exp(-value/10);
  }

  private static double norm(byte b) {
    return (b >= 0 ? (double)b : 256. + b)/255;
  }

//  private static double sim(opencv_core.Mat col, opencv_core.Mat col1) {
//    double sum = 0, sum1 = 0;
//    {
//      ByteBuffer buffer = col.createBuffer();
//      ByteBuffer buffer1 = col1.createBuffer();
//      for (int i = 0; i < 3; i++) {
//        double a = buffer.get();
//        double b = buffer1.get();
//        if (b < 0)
//          b = 256 + b;
//        if (a < 0)
//          a = 256 + a;
//        sum += a;
//        sum1 += b;
//      }
//    }
//    return MathTools.sqr((sum - sum1) / 3. / 255.);
//  }

  static double neighborhoodConvolve(int i, int j, int width, int height, double radius, CoordProcessor processor) {
    double sum = 0;
    for (int a = -(int)radius; a <= radius; a++) {
      for (int b = -(int)radius; b <= radius; b++) {
        double distance = Math.sqrt(a * a + b * b);
        if (a != 0 || b != 0 && distance < radius + MathTools.EPSILON) {
          sum += checkAndCall(i + a, j + b, width, height, distance, processor);
        }
      }
    }
    return sum;
  }

  static double checkAndCall(int i, int j, int width, int height, double distance, CoordProcessor processor) {
    if (i >= 0 && i < height && j >= 0 && j < width)
      return processor.accept(i, j, distance);
    return 0;
  }

  public interface CoordProcessor {
    double accept(int y, int x, double distance);
  }

  static class PixelGenerationModel {
    double[] sum = {0, 0, 0};
    double[] sum2 = {0, 0, 0};
    int count = 0;

    public void accept(opencv_core.Mat pixel) {
      ByteBuffer buffer = pixel.createBuffer();
      double r1 = norm(buffer.get());
      double g1 = norm(buffer.get());
      double b1 = norm(buffer.get());
      double L = 116 * labf(g1) - 16;
      double A = 500 * (labf(r1 / 0.95047) - labf(g1));
      double B = 200 * (labf(g1) - labf(b1/1.08883));
      sum[0] += L; sum2[0] += L * L;
      sum[1] += A; sum2[1] += A * A;
      sum[2] += B; sum2[2] += B * B;
      count++;
    }

    public double similarity(PixelGenerationModel other) {
      double result = 0;
      for (int i = 0; i < 3; i++) {
        result += -(mean(i) - other.mean(i)) * (mean(i) - other.mean(i)) / other.variance(i) / variance(i);
      }
      return exp(result);
    }

    private double mean(int i) {
      return sum[i] / count;
    }

    private double variance(int i) {
      return sqrt((sum2[i] - sum[i] * sum[i] / count) / (count - 1));
    }
  }
}
