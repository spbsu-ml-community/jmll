package com.spbsu.ml.methods.greedyRegion.hierarchicalCherryRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * User: solar
 * Date: 26.08.13
 * Time: 22:09
 */
@SuppressWarnings("unchecked")
public class CategoricalAggregate {
  private final BinarizedDataSet bds;
  private final BFGrid grid;
  private final Factory<AdditiveStatistics> factory;
  private final int[][] binsIndex;
  public double currentScore = 0;
  final int binsCount;
  public AdditiveStatistics inside;
  public double insideReg;

  public CategoricalAggregate(BinarizedDataSet bds, Factory<AdditiveStatistics> factory) {
    this.bds = bds;
    this.grid = bds.grid();
    this.factory = factory;
    this.binsIndex = new int[grid.rows()][];

    int current = 0;
    for (int i = 0; i < grid.rows(); i++) {
      this.binsIndex[i] = new int[grid.row(i).size() + 1];
      for (int bin = 0; bin <= grid.row(i).size(); ++bin) {
        this.binsIndex[i][bin] = current;
        ++current;
      }
    }
    this.binsCount = current;
    inside = factory.create();
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Categorical aggregate thread", -1);


  public interface FeatureVisitor<T> {
    void accept(int features, int bin, T stats);
  }
  public <T extends AdditiveStatistics> void visit(FeatureVisitor<T> visitor, final int[] points) {

    final T[][] base = (T[][]) new AdditiveStatistics[grid.rows()][];
    final CountDownLatch latch = new CountDownLatch(grid.rows());

    for (int f = 0; f < grid.rows();++f) {
      final int finalFeature = f;
      base[f] = (T[]) new AdditiveStatistics[grid.row(f).size()+1];
      exec.execute(new Runnable() {
        @Override
        public void run() {
         for (int bin =0 ; bin <= grid.row(finalFeature).size();++bin) {
           base[finalFeature][bin] = (T) factory.create();
         }
          final byte[] bins = bds.bins(finalFeature);
          for (int point: points) {
            base[finalFeature][bins[point]].append(point,1);
          }
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      // skip
    }

    for (int f = 0; f < grid.rows();++f) {
      for (int bin=0; bin <= grid.row(f).size();++bin) {
        visitor.accept(f,bin,base[f][bin]);
      }
    }
  }

}
