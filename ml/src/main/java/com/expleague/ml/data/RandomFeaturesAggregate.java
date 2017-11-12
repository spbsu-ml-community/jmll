package com.expleague.ml.data;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ThreadTools;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.data.impl.BinarizedFeature;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;


@SuppressWarnings("unchecked")
public class RandomFeaturesAggregate {

  private final Factory<AdditiveStatistics> factory;
  private final BinarizedFeatureDataSet dataSet;
  private final AdditiveStatistics[] bins;
  private final AdditiveStatistics total;
  private final FastRandom fastRandom;

  public RandomFeaturesAggregate(final Factory<AdditiveStatistics> factory,
                                 final BinarizedFeatureDataSet dataSet,
                                 final int[] indices,
                                 final double[] weights,
                                 final FastRandom fastRandom) {
    this.dataSet = dataSet;
    this.factory = factory;
    this.bins = new AdditiveStatistics[dataSet.gridHelper().binCount()];
    this.fastRandom = fastRandom;
    this.total = build(indices, weights);
  }

  public RandomFeaturesAggregate(final Factory<AdditiveStatistics> factory,
                                 final BinarizedFeatureDataSet dataSet,
                                 final AdditiveStatistics[] bins,
                                 final FastRandom fastRandom) {
    this.factory = factory;
    this.dataSet = dataSet;
    this.bins = bins;
    this.total = computeTotal(bins);
    this.fastRandom = fastRandom;
  }


  public AdditiveStatistics total() {
    return total;
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Aggregator thread", -1);

  //
  public RandomFeaturesAggregate split(final int[] points, final int[] restPoints,
                                       final double[] weights, final double[] restWeights,
                                       final boolean rebuildStochasticAggregates) {

    final AdditiveStatistics[] restBins = new AdditiveStatistics[bins.length];

    for (int i = 0; i < restBins.length; ++i) {
      restBins[i] = factory.create().append(bins[i]);
    }

    final AdditiveStatistics newTotal = build(points, weights);
    total.remove(total).append(newTotal);

    final List<BinarizedFeature> features = dataSet.features();
    final int[] offsets = dataSet.gridHelper().binOffsets();
    final CountDownLatch latch = new CountDownLatch(features.size());

    for (int findex = 0; findex < features.size(); findex++) {
      final int f = findex;
      final BinarizedFeature feature = features.get(findex);
      final FeatureBinarization binarization = feature.binarization();
      exec.execute(() -> {
        final int firstBinFeature = offsets[f];
        final int binFeatureEnd = offsets[f + 1];
        if (feature.isDeterministic() || !rebuildStochasticAggregates) {
          for (int i = firstBinFeature; i < binFeatureEnd; ++i) {
            restBins[i].remove(bins[i]);
          }
        }
        else {
          //clear first
          for (int i = firstBinFeature; i < binFeatureEnd; ++i) {
            restBins[i] = factory.create();
          }

          if (binarization.features().length > 0) {
            feature.visit(restPoints, new BinarizedFeature.FeatureVisitor() {
              @Override
              public final void accept(final int idx, final int line, final int bin, final double prob) {
                final double weight = restWeights != null ? restWeights[line] : 1.0;
                restBins[firstBinFeature + bin].append(idx, prob * weight);
              }

              @Override
              public final FastRandom random() {
                return fastRandom;
              }
            });
          }
        }
        latch.countDown();
      });
    }

    try {
      latch.await();
    }
    catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
    return new RandomFeaturesAggregate(factory, dataSet, restBins, fastRandom);
  }


  public interface BinaryFeatureVisitor<T> {
    void accept(FeatureBinarization.BinaryFeature bf, T left, T right);
  }


  public <T extends AdditiveStatistics> void visit(final BinaryFeatureVisitor<T> visitor) {
    final T total = (T) total();
    final List<BinarizedFeature> features = dataSet.features();
    final int[] offsets = dataSet.gridHelper().binOffsets();
    final CountDownLatch latch = new CountDownLatch(features.size());
    for (int findex = 0; findex < features.size(); findex++) {
      final int f = findex;
//      exec.execute(() ->
//      {
      final T left = (T) factory.create();
      final T right = (T) factory.create().append(total);
      final FeatureBinarization row = features.get(f).binarization();
      final int offset = offsets[f];
      final int rowLength = row.borders().length;
      final FeatureBinarization.BinaryFeature[] binaryFeatures = row.features();

      switch (row.type()) {
        case OneHot: {
          for (int b = 0; b < rowLength; b++) {
            final T inside = (T) factory.create().append(bins[offset + b]);
            final T outside = (T) factory.create().append(total).remove(inside);
            visitor.accept(binaryFeatures[b], inside, outside);
          }
          break;
        }
        case FloatSplit: {
          for (int b = 0; b < rowLength; b++) {
            left.append(bins[offset + b]);
            right.remove(bins[offset + b]);
            visitor.accept(binaryFeatures[b], left, right);
          }
          break;
        }
        default: {
          throw new RuntimeException("Unknown type " + row.type());
        }
      }
      latch.countDown();
    }
    ;//);
//    }

    try {
      latch.await();
    }
    catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }


  private AdditiveStatistics build(final int[] indices,
                                   final double[] weights) {

    final List<BinarizedFeature> features = dataSet.features();
    final int[] offsets = dataSet.gridHelper().binOffsets();
    final CountDownLatch latch = new CountDownLatch(features.size());

    for (int findex = 0; findex < features.size(); findex++) {
      final int f = findex;
      final BinarizedFeature feature = features.get(findex);
      final FeatureBinarization binarization = feature.binarization();
      exec.execute(() -> {
        final int offset = offsets[f];
        final int end = offsets[f + 1];

        for (int i = offset; i < end; ++i) {
          bins[i] = factory.create();
        }

        if (binarization.features().length > 0) {
          feature.visit(indices, new BinarizedFeature.FeatureVisitor() {

            @Override
            public final void accept(final int idx, final int line, final int bin, final double prob) {
              if (weights != null) {
                bins[offset + bin].append(idx, prob * weights[line]);
              }
              else {
                bins[offset + bin].append(idx, prob);
              }
            }

            @Override
            public final FastRandom random() {
              return fastRandom;
            }
          });
        }
        latch.countDown();
      });
    }

    try {
      latch.await();
    }
    catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
    return computeTotal(bins);
  }

  private AdditiveStatistics computeTotal(AdditiveStatistics[] bins) {
    final List<BinarizedFeature> features = dataSet.features();
    final int[] offsets = dataSet.gridHelper().binOffsets();

    for (int findex = 0; findex < features.size(); findex++) {
      final BinarizedFeature feature = features.get(findex);
      if (feature.binarization().features().length > 0) {
        final AdditiveStatistics stat = factory.create();
        final int firstBin = offsets[findex];
        final int nextFeatureIndex = offsets[findex + 1];
        for (int i = firstBin; i < nextFeatureIndex; ++i) {
          stat.append(bins[i]);
        }
        return stat;
      }
    }
    return factory.create();
  }
}
