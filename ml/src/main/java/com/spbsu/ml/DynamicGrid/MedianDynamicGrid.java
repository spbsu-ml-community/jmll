//package com.spbsu.ml.DynamicGrid;
//
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
//import com.spbsu.ml.BFGrid;
//import com.spbsu.ml.DynamicGrid.Interface.BinaryFeature;
//import com.spbsu.ml.DynamicGrid.Interface.DynamicGrid;
//import com.spbsu.ml.DynamicGrid.Interface.DynamicRow;
//import com.spbsu.ml.data.set.DataSet;
//import com.spbsu.ml.data.set.VecDataSet;
//import com.spbsu.ml.data.stats.OrderByFeature;
//import gnu.trove.list.array.TIntArrayList;
//
//import java.util.Arrays;
//
//
///**
// * Created by noxoomo on 22/07/14.
// */
//public class MedianDynamicGrid implements DynamicGrid {
//    private BFGrid grid;
//    private final int dim;
//    private final OrderByFeature byFeature;
//    private final TIntArrayList[] borders;
//    private final int[][] orders;
//    private final int[][] inverseOrders;
//    private final double[][] features; //sorted features
//    private final int[] levels;
//    private int[] currentLevels;
//    private boolean updating = false;
//    private boolean updated = false;
//
//    public MedianDynamicGrid(DataSet<Vec> ds, int minBinarization) {
//        dim = ((VecDataSet) ds).xdim();
//        byFeature = ds.cache().cache(OrderByFeature.class, DataSet.class);
//        borders = new TIntArrayList[dim];
//        orders = new int[dim][];
//        inverseOrders = new int[dim][];
//        levels = new int[dim];
//        currentLevels = new int[dim];
//        for (int f = 0; f < dim; ++f) {
//            borders[f] = new TIntArrayList();
//            borders[f].add(ds.length());
//            final ArrayPermutation permutation = byFeature.orderBy(f);
//            orders[f] = permutation.direct();
//            inverseOrders[f] = permutation.reverse();
//        }
//        features = new double[dim][ds.length()];
//        for (int f = 0; f < dim; ++f) {
//            int[] order = orders[f];
//            for (int i = 0; i < order.length; ++i)
//                features[f][i] = ds.at(order[i]).get(f);
//        }
//
//        for (int f = 0; f < dim; ++f) {
//            int level = 0;
//            double[] feature = features[f];
//            for (int i = 1; i < feature.length; ++i)
//                if (feature[i] != feature[i - 1])
//                    ++level;
//            levels[f] = level;
//        }
//
//        beginUpdate();
//        for (int i = 0; i < minBinarization - 1; ++i) {
//            for (int f = 0; f < dim; ++f)
//                growRow(f);
//        }
//        endUpdate();
//    }
//
//    @Override
//    public BFGrid getGrid() {
//        return grid;
//    }
//
//    @Override
//    public boolean growRow(int rowIndex) {
//        if (currentLevels[rowIndex] == levels[rowIndex])
//            return false;
//        double[] feature = features[rowIndex];
//        TIntArrayList border = borders[rowIndex];
//        double bestScore = 0;
//        int bestSplit = -1;
//        for (int i = 0; i < border.size(); ++i) {
//            int start = i > 0 ? border.get(i - 1) : 0;
//            int end = border.get(i);
//            double median = feature[start + (end - start) / 2];
//            int split = Math.abs(Arrays.binarySearch(feature, start, end, median));
//            while (split > 0 && Math.abs(feature[split] - median) < 1e-9) // look for first less then median value
//                split--;
//            if (Math.abs(feature[split] - median) > 1e-9) split++;
//            final double scoreLeft = Math.log(end - split) + Math.log(split - start);
//            if (split > 0 && scoreLeft > bestScore) {
//                bestScore = scoreLeft;
//                bestSplit = split;
//            }
//            while (++split < end && Math.abs(feature[split] - median) < 1e-9)
//                ; // first after elements with such value
//            final double scoreRight = Math.log(end - split) + Math.log(split - start);
//            if (split < end && scoreRight > bestScore) {
//                bestScore = scoreRight;
//                bestSplit = split;
//            }
//            if (bestSplit < 0)
//                return false;
//            border.add(bestSplit);
//            border.sort();
//        }
//        if (!updating)
//            recalcGrid();
//        else updated = true;
//
//        return true;
//    }
//
//    private void recalcGrid() {
//        BFGrid.BFRow[] rows = new BFGrid.BFRow[dim];
//        int bfCount = 0;
//        for (int f = 0; f < rows.length; ++f) {
//            double[] dborders = new double[borders[f].size() - 1];
//            for (int b = 0; b < borders[f].size() - 1; b++) {
//                dborders[b] = features[f][borders[f].get(b) - 1];
//            }
//            rows[f] = new BFGrid.BFRow(bfCount, f, dborders);
//            bfCount += dborders.length;
//        }
//        this.grid = new BFGrid(rows);
//    }
//
//    @Override
//    public boolean grow(int f) {
//        BFGrid.BinaryFeature bf = grid.bf(f);
//        return growRow(bf.row().origFIndex);
//
//    }
//
//    @Override
//    public int rows() {
//        return 0;
//    }
//
//    @Override
//    public int size() {
//        return bfs.si
//    }
//
//    @Override
//    public boolean isActive(int feature) {
//        BFGrid.BinaryFeature bf = grid.bf(feature);
//        return bf.origFIndex < (bf.row().size() - 1);
//    }
//
//    @Override
//    public DynamicRow row(int feature) {
//        return null;
//    }
//
//    @Override
//    public void binarize(Vec x, int[] folds) {
//
//    }
//
//    @Override
//    public BinaryFeature bf(int index) {
//        return null;
//    }
//
//    @Override
//    public DynamicRow nonEmptyRow() {
//        return null;
//    }
//
//    @Override
//    public boolean addSplit(int feature) {
//        return false;
//    }
//
//    @Override
//    public void queueSplit(BinaryFeature bf) {
//
//    }
//
//
//    @Override
//    public void acceptQueue() {
//
//    }
//
//
//    @Override
//    public int maxBinsCount(int f) {
//        return levels[f];
//    }
//
//    @Override
//    public void beginUpdate() {
//        updating = true;
//        updated = false;
//    }
//
//    @Override
//    public void endUpdate() {
//        updating = false;
//        if (updated)
//            recalcGrid();
//    }
//}
//
