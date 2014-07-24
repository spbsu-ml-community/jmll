package com.spbsu.ml.DynamicGrid.Impl;

import com.spbsu.ml.DynamicGrid.Interface.BinaryFeature;
import com.spbsu.ml.DynamicGrid.Interface.DynamicGrid;
import com.spbsu.ml.DynamicGrid.Interface.DynamicRow;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

/**
 * Created by noxoomo on 23/07/14.
 */
public class MedianRow implements DynamicRow {
    private final double eps = 1e-9;
    private final int origFIndex;
    private DynamicGrid grid = null;
    private final double[] feature;
    //    private final TIntArrayList borders = new TIntArrayList();
    private final ArrayList<BinaryFeatureImpl> bfs = new ArrayList<>();
    private final int levels;


    public MedianRow(DynamicGrid grid, double[] feature, int origFIndex, int minSplits) {
        this.origFIndex = origFIndex;
        this.feature = feature;
        this.grid = grid;
        int lvl = 0;
        for (int i = 1; i < feature.length; ++i)
            if (feature[i] != feature[i - 1])
                ++lvl;
        this.levels = lvl;
        bfs.add(new BinaryFeatureImpl(this, origFIndex, Double.POSITIVE_INFINITY, feature.length));
//        borders.add(feature.length);
        for (int i = 0; i < minSplits; ++i)
            addSplit();
        for (BinaryFeature bf : bfs)
            bf.setActive(true);
        addSplit();
    }

    public MedianRow(double[] feature, int origFIndex) {
        this(null, feature, origFIndex, 1);
    }

    public MedianRow(double[] feature, int origFIndex, int minSplits) {
        this(null, feature, origFIndex, minSplits);
    }

    public ArrayList<BinaryFeatureImpl> features() {
        return bfs;
    }

    @Override
    public int origFIndex() {
        return origFIndex;
    }


    @Override
    public int size() {
        return bfs.size() - 1;
    }

    @Override
    public DynamicGrid grid() {
        return grid;
    }

    private static Random rand = new Random();

    @Override
    public boolean addSplit() {
        if (bfs.size() >= levels + 1)
            return false;
        double bestScore = 0;
        double diff = 0;
        int bestSplit = -1;
        TIntArrayList bestSplits = new TIntArrayList();
        for (int i = 0; i < bfs.size(); ++i) {
            int start = i > 0 ? bfs.get(i - 1).borderIndex : 0;
            int end = bfs.get(i).borderIndex;
            double median = feature[start + (end - start) / 2];
            int split = Math.abs(Arrays.binarySearch(feature, start, end, median));
            while (split > 0 && Math.abs(feature[split] - median) < eps) // look for first less then median value
                split--;
            if (Math.abs(feature[split] - median) > 1e-9) split++;
            final double scoreLeft = Math.log(end - split) + Math.log(split - start);
            if (split > 0) {
                if (scoreLeft > bestScore + 1e-8) {
                    bestScore = scoreLeft;
                    diff = (end-start+1)* Math.log( (end - start + 1.0) / feature.length) - (end - split) * Math.log( (end - split)*1.0 / feature.length) - (split - start + 1) * Math.log( (split - start + 1.0) / feature.length);
                    diff /= feature.length;
                    bestSplit = split;
                    bestSplits.clear();
                    bestSplits.add(bestSplit);
                } else if (Math.abs(scoreLeft - bestScore) < 1e-8) {
                    bestSplits.add(split);
                }
            }
            while (++split < end && Math.abs(feature[split] - median) < eps)
                ; // first after elements with such value
            final double scoreRight = Math.log(end - split) + Math.log(split - start);
            if (split < end) {
                if (scoreRight > bestScore + 1e-8) {
                    bestScore = scoreRight;
                    diff = (end-start+1)* Math.log( (end - start + 1.0) / feature.length) - (end - split) * Math.log( (end - split)*1.0 / feature.length) - (split - start + 1) * Math.log( (split - start + 1.0) / feature.length);
                    diff /= feature.length;
                    bestSplit = split;
                    bestSplits.clear();
                    bestSplits.add(bestSplit);

                } else if (Math.abs(scoreRight - bestScore) < 1e-8) {
                    bestSplits.add(split);
                }
            }
        }
        if (bestSplit < 0)
            return false;
        bestSplit = bestSplits.get(rand.nextInt(bestSplits.size()));
        BinaryFeatureImpl newBF = new BinaryFeatureImpl(this, origFIndex, feature[bestSplit - 1], bestSplit);
        bfs.add(newBF);
        newBF.setRegScore(diff);

        Collections.sort(bfs, BinaryFeatureImpl.borderComparator);
        for (int i = 0; i < bfs.size(); ++i) {
            bfs.get(i).setBinNo(i);
        }
        return true;
    }

    @Override
    public boolean empty() {
        return size() == 0;
    }

    @Override
    public BinaryFeature bf(int binNo) {
        return bfs.get(binNo);
    }

    @Override
    public void setOwner(DynamicGrid grid) {
        this.grid = grid;
    }

    @Override
    public int bin(double value) {
        int index = 0;
        while (index < (bfs.size() - 1) && value > bfs.get(index).condition)
            index++;
        return index;
    }
}
