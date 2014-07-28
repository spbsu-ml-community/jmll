package com.spbsu.ml.data;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * User: solar
 * Date: 26.08.13
 * Time: 22:09
 */
@SuppressWarnings("unchecked")
public class Aggregate {
    private final BinarizedDataSet bds;
    int transferRowSize;
    private final BFGrid grid;
    public final AdditiveStatistics[] bins;
    private final Factory<AdditiveStatistics> factory;
    private volatile AdditiveStatistics total;

    public Aggregate(BinarizedDataSet bds, Factory<AdditiveStatistics> factory, int[] points) {
        this.bds = bds;
        this.grid = bds.grid();
        int maxRow = 0;
        for (int i = 0; i < grid.rows(); i++) {
            maxRow = Math.max(maxRow, grid.row(i).size()) + 1;
        }
        transferRowSize = maxRow;
        this.bins = new AdditiveStatistics[transferRowSize * grid.rows()];
        for (int i = 0; i < bins.length; i++) {
            bins[i] = factory.create();

        }
        this.factory = factory;
        build(points);
    }

    public AdditiveStatistics combinatorForFeature(int bf) {
        final AdditiveStatistics result = factory.create();

        final BFGrid.BFRow row = grid.bf(bf).row();
        final int binNo = grid.bf(bf).binNo;
        final int offset = row.origFIndex * transferRowSize;
        for (int b = 0; b <= binNo; b++) {
            result.append(bins[offset + b]);
        }
        return result;
    }

    public AdditiveStatistics total() {
//    if (total == null) { // calculating total by non empty row
        AdditiveStatistics myTotal = factory.create();
        final BFGrid.BFRow row = grid.nonEmptyRow();
        final int offset = row.origFIndex * transferRowSize;
        final AdditiveStatistics[] myBins = bins;
        for (int b = 0; b <= row.size(); b++) {
            myTotal.append(myBins[offset + b]);
        }
        return myTotal;
//      total = myTotal;
//    }

//    return total;
    }

    private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Aggregator thread", -1);

    public void remove(final Aggregate aggregate) {
        final AdditiveStatistics[] my = bins;
        final AdditiveStatistics[] other = aggregate.bins;
        for (int i = 0; i < bins.length; i++) {
            my[i].remove(other[i]);
        }
//    total.remove(aggregate.total());
    }

    public void append(final Aggregate aggregate) {
        final AdditiveStatistics[] my = bins;
        final AdditiveStatistics[] other = aggregate.bins;
        for (int i = 0; i < bins.length; i++) {
            my[i].append(other[i]);
        }
//    total.remove(aggregate.total());
    }

    public void removeTest(final Aggregate aggregate, final Aggregate groundTruth) {
        final AdditiveStatistics[] my = bins;
        final AdditiveStatistics[] other = aggregate.bins;
        boolean equals = true;
        for (int i = 0; i < bins.length; i++) {
            my[i].remove(other[i]);
            L2.MSEStats first = (L2.MSEStats) (((WeightedLoss.Stat) my[i])).inside;
            L2.MSEStats second = (L2.MSEStats) (((WeightedLoss.Stat) groundTruth.bins[i])).inside;
            if (Math.abs(first.sum - second.sum) > 1e-9 || Math.abs(first.sum2 - second.sum2) > 1e-9) {
                equals = false;
                break;

            }
        }
//    total.remove(aggregate.total());
    }


    public interface SplitVisitor<T> {
        void accept(BFGrid.BinaryFeature bf, T left, T right);
    }

    public <T extends AdditiveStatistics> void visit(SplitVisitor<T> visitor) {
        final T total = (T) total();

        for (int f = 0; f < grid.rows(); f++) {
            final T left = (T) factory.create();
            final T right = (T) factory.create().append(total);
            final BFGrid.BFRow row = grid.row(f);
            for (int b = 0; b < row.size(); b++) {
                left.append(bins[row.origFIndex * transferRowSize + b]);
                right.remove(bins[row.origFIndex * transferRowSize + b]);
                visitor.accept(row.bf(b), left, right);
            }
        }
    }

    private void build(final int[] indices) {
        final CountDownLatch latch = new CountDownLatch(grid.rows());
        for (int findex = 0; findex < grid.rows(); findex++) {
            final int finalFIndex = findex;
            exec.execute(new Runnable() {
                @Override
                public void run() {
                    final byte[] bin = bds.bins(finalFIndex);
                    final int offset = finalFIndex * transferRowSize;
                    if (!grid.row(finalFIndex).empty()) {
//                        for (int i : indices) {
//                            bins[offset + bin[i]].append(i, 1);
//                        }
                        final int length = 4 * (indices.length / 4);
                        final AdditiveStatistics[] binsLocal = bins;
                        final int[] indicesLocal = indices;
                        for (int i = 0; i < length; i += 4) {
                            final int idx1 = indicesLocal[i];
                            final int idx2 = indicesLocal[i + 1];
                            final int idx3 = indicesLocal[i + 2];
                            final int idx4 = indicesLocal[i + 3];
                            final AdditiveStatistics bin1 = binsLocal[offset + bin[idx1]];
                            final AdditiveStatistics bin2 = binsLocal[offset + bin[idx2]];
                            final AdditiveStatistics bin3 = binsLocal[offset + bin[idx3]];
                            final AdditiveStatistics bin4 = binsLocal[offset + bin[idx4]];
                            bin1.append(idx1, 1);
                            bin2.append(idx2, 1);
                            bin3.append(idx3, 1);
                            bin4.append(idx4, 1);
                        }
                        for (int i = 4 * (indicesLocal.length / 4); i < indicesLocal.length; i++) {
                            binsLocal[offset + bin[indicesLocal[i]]].append(indicesLocal[i], 1);
                        }
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
    }
}
