package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.impl.RankedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.LinearRegion;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by au-rikka on 29.04.17.
 */
public class GreedyTDProbRegion<Loss extends WeightedLoss<? extends L2>> extends VecOptimization.Stub<Loss> {
    protected final BFGrid grid;
    private final int depth;
    private final double lambda;
    private final double beta;
    private final double alpha;
    private final Random rnd = new Random();
    private int[][] ordered;

    public GreedyTDProbRegion(final BFGrid grid,
                              final int depth,
                              final double lambda,
                              final double beta,
                              final double alpha) {
        this.grid = grid;
        this.depth = depth;
        this.lambda = lambda;
        this.beta = beta; // false_negative
        this.alpha = alpha; // false_positive
    }

    @Override
    public LinearRegion fit(final VecDataSet learn, final Loss loss) {

        Vec target = VecTools.copy(loss.target());
        double weights[] = extractWeights(loss);
        double betas[] = new double[depth];

        ordered = new int[learn.xdim()][];

        final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(depth);
        final List<Boolean> mask = new ArrayList<>();
        final boolean[] usedBF = new boolean[grid.size()];

        int[] points = learnPoints(loss, learn);

        final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

        WeightedLoss.Stat stat = loss.statsFactory().create();
        for (int i = 0; i < points.length; i++) {
            stat.append(points[i], 1);
        }

        betas[0] = loss.bestIncrement(stat);
        for (int i = 0; i < points.length; i++) {
            target.adjust(points[i], -betas[0]);
        }

        int betasSize = 1;

        double previousScore = -1;

        for (int level = 1; level < depth; level++) {
            final L2 curLoss = DataTools.newTarget(loss.base().getClass(), target, learn);
            final WeightedLoss<L2> wCurLoss = new WeightedLoss<L2>(curLoss, weights);

            final BFOptimizationSimpleRegion current = new BFOptimizationSimpleRegion(bds, wCurLoss, points);

            double newScore = getScore(current.total(), 0);
//            if (previousScore != -1 && newScore > previousScore) {
//                break;
//            }

            System.out.println(previousScore);

            betasSize = level + 1;

            previousScore = newScore;

            final double[] scores = new double[grid.size()];
            final double[] solution = new double[grid.size()];
            final boolean[] isRight = new boolean[grid.size()];

            current.visitAllSplits((bf, left, right) -> {
                if (usedBF[bf.bfIndex]) {
                    scores[bf.bfIndex] = Double.POSITIVE_INFINITY;
                    solution[bf.bfIndex] = -1;
                } else {
                    final double leftWeight = weight(left);
                    final double rightWeight = weight(right);
                    final double minExcluded = Math.min(leftWeight, rightWeight);

                    final double leftScore;
                    double leftBeta = -1;

                    double rightBeta = -1;
                    final double rightScore;

                    if (minExcluded > 3) {
                        double p[] = getWeightsProb(bf, false, learn, weights.length);

                        Pair<Double, Double> leftStat = getStat(p, target, weights);

                        leftBeta = leftStat.first;
                        leftScore = leftStat.second;

                    } else {
                        leftScore = Double.POSITIVE_INFINITY;
                    }

                    if (minExcluded > 3) {
                        double p[] = getWeightsProb(bf, true, learn, weights.length);

                        Pair<Double, Double> rightStat = getStat(p, target, weights);
                        rightBeta = rightStat.first;
                        rightScore = rightStat.second;
                    } else {
                        rightScore = Double.POSITIVE_INFINITY;
                    }
                    scores[bf.bfIndex] = leftScore > rightScore ? rightScore : leftScore;
                    isRight[bf.bfIndex] = leftScore > rightScore;
                    solution[bf.bfIndex] = isRight[bf.bfIndex] ? rightBeta : leftBeta;
                }
            });

            final int bestSplit = ArrayTools.min(scores);
            if (bestSplit < 0)
                break;

            final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
            final boolean bestSplitMask = isRight[bestSplitBF.bfIndex];

            betas[level] = solution[bestSplit];

            if (level < (depth - 1)) {

//                current.split(bestSplitBF, bestSplitMask);
//                points = current.getPoints();


//                if (previousScore <= currentScore) {
//                    //                   betasSize -= 1;
//                    break;
//                }

                double[] p = getWeightsProb(bestSplitBF, bestSplitMask, learn, weights.length);

                for (int i = 0; i < points.length; i++) {
                    target.adjust(points[i], -p[i]*betas[level]);
                }

                points = sample(p, weights);

            }

            conditions.add(bestSplitBF);
            usedBF[bestSplitBF.bfIndex] = true;
            mask.add(bestSplitMask);
        }

        final boolean[] masks = new boolean[mask.size()];
        for (int i = 0; i < masks.length; i++) {
            masks[i] = mask.get(i);
        }

//
        final double bias = betas[0];
        final double[] values = new double[betasSize - 1];
        System.arraycopy(betas, 1, values, 0, values.length);

        return new LinearRegion(conditions, masks, bias, values);
    }

    private double[] extractWeights(Loss loss) {
        double[] weights = new double[loss.dim()];
        for (int i = 0; i < loss.dim(); i++) {
            weights[i] = loss.weight(i);
        }
        return weights;
    }

    private int[] learnPoints(Loss loss, VecDataSet ds) {
        if (loss instanceof WeightedLoss) {
            return ((WeightedLoss) loss).points();
        } else
            return ArrayTools.sequence(0, ds.length());
    }

    private double weight(final AdditiveStatistics stat) {
        if (stat instanceof L2.MSEStats) {
            return ((L2.MSEStats) stat).weight;
        } else if (stat instanceof WeightedLoss.Stat) {
            return weight(((WeightedLoss.Stat) stat).inside);
        } else {
            throw new RuntimeException("error");
        }
    }

    private double mean(final AdditiveStatistics stat) {
        if (stat instanceof L2.MSEStats) {
            L2.MSEStats curStat = (L2.MSEStats) stat;
            return curStat.sum / (curStat.weight + 1);
        } else if (stat instanceof WeightedLoss.Stat) {
            return mean(((WeightedLoss.Stat) stat).inside);
        } else {
            throw new RuntimeException("error");
        }
    }

//    private double getScore(Vec target, int[] points) {
//        double score = 0;
//        for (int i = 0; i < points.length; i++) {
//            score += target.get(points[i]);
//        }
//        return score*MathTools.sqr(points.length/(points.length - 1));
//    }

    private double getScore(AdditiveStatistics stat, double v) {
        L2.MSEStats statL2;
        if (stat instanceof WeightedLoss.Stat) {
            statL2 = (L2.MSEStats) ((WeightedLoss.Stat) stat).inside;
        } else if (stat instanceof L2.MSEStats){
            statL2 = (L2.MSEStats) stat;
        } else {
            return -1;
        }
        return (statL2.sum2 - 2 * v * statL2.sum + v * v * statL2.weight) * MathTools.sqr(statL2.weight / (statL2.weight - 1));
    }

    private double[] getWeightsProb(final BFGrid.BinaryFeature bestSplitBF, boolean bestSplitMask, VecDataSet learn,
                                    final int size) {
        if (ordered[bestSplitBF.findex] == null) {
            ordered[bestSplitBF.findex] = (new RankedDataSet(learn)).byFeature.orderBy(bestSplitBF.findex).direct();
        }
        int lastTaken;
        double p[] = new double[size];
        if (bestSplitMask) {
            lastTaken = ordered[bestSplitBF.findex].length - 1;
            for (int i = lastTaken; i >= 0; i--) {
                if (bestSplitBF.value(learn.at(ordered[bestSplitBF.findex][i]))) {
                    lastTaken = i;
                }
            }
        } else {
            lastTaken = 0;
            for (int i = 0; i < ordered[bestSplitBF.findex].length - 1; i++) {
                if (!bestSplitBF.value(learn.at(ordered[bestSplitBF.findex][i]))) {
                    lastTaken = i;
                }
            }
        }

        for (int i = 0; i < ordered[bestSplitBF.findex].length; i++) {
            if (bestSplitBF.value(learn.at(ordered[bestSplitBF.findex][i])) == bestSplitMask) {
                p[ordered[bestSplitBF.findex][i]] = (1 - Math.exp(-this.alpha*(Math.abs(i - lastTaken) + 1)));
            } else {
                p[ordered[bestSplitBF.findex][i]] = Math.exp(-this.beta*(Math.abs(i - lastTaken)));
            }
        }

        return p;
    }

    private Pair<Double, Double> getStat(double[] p, Vec target, double[] w) {
        double py = 0;
        double py2 = 0;
        double psum = 0;

        double score = 0;
        double beta = 0;

        for (int i = 0; i < p.length; i++) {
            py2 += p[i]*w[i]*target.at(i)*target.get(i);
            py += w[i]*target.at(i)*p[i];
            psum += w[i]*p[i];
        }

        beta = py/psum;
        score = py2 - py*py/psum;

        return new Pair<>(beta, score);
    }

    private int[] sample(double p[], double weights[]) {
        final TIntArrayList result = new TIntArrayList(weights.length);
        for (int i = 0; i < weights.length; i++) {
            int cntPnts = (int)weights[i];
            for (int j = 0; j < cntPnts; j++) {
                double v = rnd.nextDouble();
                if (v > p[i]) {
                    weights[i] -= 1;
                }
            }
            if (weights[i] > MathTools.EPSILON) {
                result.add(i);
            }
        }
        return result.toArray();
    }

//    private WeightedLoss.Stat totalStat(final WeightedLoss loss, final ) {
//        WeightedLoss.Stat stat = (WeightedLoss.Stat) loss.statsFactory().create();
//        for (int i = 0; i < ; i++) {
//            stat.append(points[i], 1);
//        }
//    }
}
