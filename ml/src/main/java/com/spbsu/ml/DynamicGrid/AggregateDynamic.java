package com.spbsu.ml.DynamicGrid;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.DynamicGrid.Interface.BinaryFeature;
import com.spbsu.ml.DynamicGrid.Interface.DynamicGrid;
import com.spbsu.ml.DynamicGrid.Interface.DynamicRow;
import com.spbsu.ml.data.impl.BinarizedDynamicDataSet;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

@SuppressWarnings("unchecked")
public class AggregateDynamic {
    private final BinarizedDynamicDataSet bds;
    private final DynamicGrid grid;
    private final AdditiveStatistics[][] bins;
    private final Factory<AdditiveStatistics> factory;
    private final int[] points;

    public AggregateDynamic(BinarizedDynamicDataSet bds, Factory<AdditiveStatistics> factory, int[] points) {
        this.points = points;
        this.bds = bds;
        this.grid = bds.grid();
        this.bins = new AdditiveStatistics[grid.rows()][];
        for (int feat = 0; feat < bins.length; feat++) {
            bins[feat] = new AdditiveStatistics[0];
        }
        this.factory = factory;
        rebuild(points, ArrayTools.sequence(0, grid.rows()));
    }

    public AdditiveStatistics combinatorForFeature(BinaryFeature bf) {
        final AdditiveStatistics result = factory.create();

        final DynamicRow row = bf.row();
        final int binNo = bf.binNo();
        final int origFIndex = row.origFIndex();
        for (int b = 0; b <= binNo; b++) {
            result.append(bins[origFIndex][b]);
        }
        return result;
    }

    public AdditiveStatistics total() {
//    if (total == null) { // calculating total by non empty row
        AdditiveStatistics myTotal = factory.create();
        final DynamicRow row = grid.nonEmptyRow();
        final AdditiveStatistics[][] myBins = bins;
        for (int bin = 0; bin < bins[row.origFIndex()].length; ++bin) {
            myTotal.append(myBins[row.origFIndex()][bin]);
        }
        return myTotal;
//      total = myTotal;
//    }

//    return total;
    }

    private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Aggregator thread", -1);

    public void remove(final AggregateDynamic aggregate) {
        final AdditiveStatistics[][] my = bins;
        final AdditiveStatistics[][] other = aggregate.bins;
        for (int i = 0; i < bins.length; i++) {
            for (int j = 0; j < bins[i].length; ++j)
                my[i][j].remove(other[i][j]);
        }
//    total.remove(aggregate.total());
    }

    public interface SplitVisitor<T> {
        void accept(BinaryFeature bf, T left, T right);

    }

    public <T extends AdditiveStatistics> void visit(SplitVisitor<T> visitor) {
        final T total = (T) total();

        for (int f = 0; f < grid.rows(); f++) {
            final T left = (T) factory.create();
            final T right = (T) factory.create().append(total);
            final DynamicRow row = grid.row(f);
            for (int b = 0; b < row.size(); b++) {
                left.append(bins[row.origFIndex()][b]);
                right.remove(bins[row.origFIndex()][b]);
                visitor.accept(row.bf(b), left, right);
            }
        }
    }

    public void rebuild(int... features) {
        rebuild(this.points, features);
    }

    private void rebuild(final int[] indices, int... features) {
        final CountDownLatch latch = new CountDownLatch(features.length);
        for (int findex : features) {
            final int finalFIndex = findex;
            exec.execute(new Runnable() {
                @Override
                public void run() {
                    final int[] bin = bds.bins(finalFIndex);
                    if (!grid.row(finalFIndex).empty()) {
                        final int length = 4 * (indices.length / 4);
                        final AdditiveStatistics[] binsLocal = new AdditiveStatistics[grid.row(finalFIndex).size() + 1];
                        for (int i = 0; i < binsLocal.length; ++i)
                            binsLocal[i] = factory.create();
                        final int[] indicesLocal = indices;
                        for (int i = 0; i < length; i += 4) {
                            final int idx1 = indicesLocal[i];
                            final int idx2 = indicesLocal[i + 1];
                            final int idx3 = indicesLocal[i + 2];
                            final int idx4 = indicesLocal[i + 3];
                            final AdditiveStatistics bin1 = binsLocal[bin[idx1]];
                            final AdditiveStatistics bin2 = binsLocal[bin[idx2]];
                            final AdditiveStatistics bin3 = binsLocal[bin[idx3]];
                            final AdditiveStatistics bin4 = binsLocal[bin[idx4]];
                            bin1.append(idx1, 1);
                            bin2.append(idx2, 1);
                            bin3.append(idx3, 1);
                            bin4.append(idx4, 1);
                        }
                        for (int i = 4 * (indicesLocal.length / 4); i < indicesLocal.length; i++) {
                            binsLocal[bin[i]].append(indicesLocal[i], 1);
                        }
                        bins[finalFIndex] = binsLocal;
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
