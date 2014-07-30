package com.spbsu.ml.DynamicGrid.toRun;


import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.DynamicGrid.Trees.GreedyObliviousTreeDynamic;
import com.spbsu.ml.DynamicGrid.Trees.GreedyObliviousTreeDynamic2;
import com.spbsu.ml.DynamicGrid.Trees.GreedyObliviousTreeDynamic3;
import com.spbsu.ml.Func;
import com.spbsu.ml.GridTools;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LogL2;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.RandomForest;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.trees.DynamicGrid.Trees.GreedyObliviousTreeDynamicOld;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TLongArrayList;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static com.spbsu.ml.Utils.stats;


/**
 * Created by noxoomo on 11/07/14.
 */

public class Main {
    static FastRandom random = new FastRandom(12);
    private static int iterations = 1500;
    private static double step = 0.004;
    private static double lambda = 0.25;
    private static int minBinFactor = 1;
    private static int restartFactor = 210;
    private static int restartIterations = 1;
    private static int maxBinFactor = 50;
    private static int treesCount = 3;
    private static int worksFactor = 2;
    private static double learnStep = 0.001;
    private static int learnIterations = 4000;
    private static int aggregateIterations = 1000;
    private static int warmUpIterations = 20;
    private static boolean randomNoise = true;


    public static void main(String[] args) {

        try {
            String learnPath = randomNoise ? "/Users/noxoomo/Projects/jmll2/jmll/ml/src/test/data/featuresWithNoise.txt"
                    : "/Users/noxoomo/Projects/jmll2/jmll/ml/src/test/data/features.txt.gz";
            String testPath = randomNoise ? "/Users/noxoomo/Projects/jmll2/jmll/ml/src/test/data/featuresTestWithNoise.txt"
                    : "/Users/noxoomo/Projects/jmll2/jmll/ml/src/test/data/featuresTest.txt.gz";
            Pool<?> learn = DataTools.loadFromFeaturesTxt(learnPath);
            Pool<?> validate = DataTools.loadFromFeaturesTxt(testPath);
            System.err.println(String.format("Learn size %d\nValidation size: %d", learn.data().length(), validate.data().length()));

            System.err.println(String.format("Step size: %f", step));
            System.err.println(String.format("Lambda: %f", lambda));
            System.err.println(String.format("minBinFactor: %d", minBinFactor));


            switch (args[0]) {
                case "classic": {
                    System.err.println("Classical boosting on Trees\n\n");
                    runBoosting(learn, validate, new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(learn.vecData(), 32), 6), random), 8);
                    break;
                }

                case "classic-rf": {
                    System.err.println("Classical boosting with rf on Trees\n\n");
                    runBoosting(learn, validate, new RandomForest(new GreedyObliviousTree(GridTools.medianGrid(learn.vecData(), 32), 6), random, treesCount), 8);
                    break;
                }
//
//                case "RF": {
//                    System.err.println("Random Forest Greedy Grid\n\n");
//                    GreedyDynamicGridObliviousTree tree = new GreedyDynamicGridObliviousTree<>(learn.vecData(), 6, minBinFactor, maxBinFactor, lambda);
//                    tree.setLog(true);
//                    RandomForest rf = new RandomForest(tree, random, treesCount);
//                    runBoosting(learn, validate, rf);
//                    break;
//                }
                case "dynamic": {
                    System.err.println("Greedy dynamic grid Grid\n\n");
                    runBoostingDynamic(learn, validate, 16, true);
                    break;
                }


                case "dynamic2": {
                    System.err.println("Greedy dynamic grid Grid\n\n");
                    runBoostingDynamic2(learn, validate, 4, true);
                    break;
                }


                case "dynamic3-bootstrap": {
                    System.err.println("Greedy dynamic grid Grid\n\n");
                    runBoostingDynamic3(learn, validate, 4, true);
                    break;
                }

                case "dynamic2-bootstrap": {
                    System.err.println("Greedy dynamic grid Grid\n\n");
                    runBoostingDynamic2(learn, validate, 4, false);
                    break;
                }


                case "dynamic-bootstrap": {
                    System.err.println("Greedy dynamic grid Grid\n\n");
                    runBoostingDynamic(learn, validate, 4, false);
                    break;
                }


                case "dynamic-bootstrap-restart": {
                    System.err.println("Greedy dynamic grid Grid\n\n");
                    runBoostingDynamicWithRestart(learn, validate, 4, false);
                    break;
                }

                case "dynamicOld-rf": {
                    System.err.println("Greedy dynamic grid Grid\n\n");
                    runBoostingDynamicOld(learn, validate, 1, true);
                    break;
                }

                case "dynamicOld-bootstrap": {
                    System.err.println("Greedy dynamic grid Grid\n\n");
                    runBoostingDynamicOld(learn, validate, 1, false);
                    break;
                }

                case "AggregateTest": {
                    TLongArrayList scores = new TLongArrayList();
                    List<Trans> results = new ArrayList(aggregateIterations);

                    //warm up
                    for (int i = 0; i < warmUpIterations; ++i) {
                        final VecOptimization weak = new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(learn.vecData(), 32), 6), new FastRandom(random.nextLong()));
                        results.add(weak.fit(learn.vecData(), (L2) learn.target(LogL2.class)));
                    }

                    for (int i = 0; i < aggregateIterations; ++i) {
                        final long start = System.currentTimeMillis();
                        final VecOptimization weak = new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(learn.vecData(), 32), 6), new FastRandom(random.nextLong()));
                        final long end = System.currentTimeMillis();
                        results.add(weak.fit(learn.vecData(), (L2) learn.target(LogL2.class)));
                        scores.add(end - start);
                    }
                    System.out.println(mkString(scores.toArray()));
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private static void runBoostingDynamicOld(Pool learn, Pool validate, int tries, boolean rf) {
        TDoubleArrayList scores = new TDoubleArrayList();
        for (int tr = 0; tr < tries; ++tr) {
//            final GreedyObliviousTreeDynamic tree = new GreedyObliviousTreeDynamic(learn.vecData(), 6, lambda, minBinFactor);
            final GreedyObliviousTreeDynamicOld tree = new GreedyObliviousTreeDynamicOld(learn.vecData(), 6, minBinFactor, maxBinFactor);
//            tree.stopGrowing();
//            VecOptimization weak =  new BootstrapOptimization(tree, new FastRandom(330));
            VecOptimization weak = rf ? new RandomForest(tree, new FastRandom(random.nextLong()), treesCount)
                    : new BootstrapOptimization(tree, new FastRandom(random.nextLong()));//new RandomForest(tree, random, 4);
//        VecOptimization weak = n

            GradientBoosting<L2> boosting = new GradientBoosting<>(weak, iterations, step);
            final Action counter = new ProgressHandler() {
                int index = 0;

                @Override
                public void invoke(Trans partial) {
                    System.out.println("\n\nCurrent binarization: " + mkString(tree.hist()));
                    System.out.print(index++);
                }
            };
//


            final L2 target = (L2) learn.target(LogL2.class);
            final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), target);
            final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), (L2) validate.target(LogL2.class));
            boosting.addListener(counter);
            boosting.addListener(learnListener);
            boosting.addListener(validateListener);
            boosting.fit(learn.vecData(), (L2) learn.target(LogL2.class));
            scores.add(validateListener.min);
        }

        double stat[] = stats(scores);
        System.out.println(String.format("Score stats are %f ± %f, min: %f, max: %f\n\n", stat[0], stat[1], stat[2], stat[3]));
    }

    private static void runBoostingDynamicBootstrap(Pool learn, Pool validate) {
        runBoostingDynamic(learn, validate, 1, false);
    }

    private static void runBoostingDynamicRF(Pool learn, Pool validate) {
        runBoostingDynamic(learn, validate, 1, true);
    }


    private static void runBoostingDynamic(Pool learn, Pool validate, int tries, boolean rf) {
        TDoubleArrayList scores = new TDoubleArrayList();
        for (int tr = 0; tr < tries; ++tr) {
            final GreedyObliviousTreeDynamic tree = new GreedyObliviousTreeDynamic(learn.vecData(), 6, lambda, minBinFactor);
//            tree.stopGrowing();
//            VecOptimization weak =  new BootstrapOptimization(tree, new FastRandom(330));
            VecOptimization weak = rf ? new RandomForest(tree, new FastRandom(random.nextLong()), treesCount)
                    : new BootstrapOptimization(tree, new FastRandom(random.nextLong()));//new RandomForest(tree, random, 4);
//        VecOptimization weak = n

            GradientBoosting<L2> boosting = new GradientBoosting<>(weak, iterations, step);
            final Action counter = new ProgressHandler() {
                int index = 0;

                @Override
                public void invoke(Trans partial) {
                    System.out.println("\n\nCurrent binarization: " + mkString(tree.hist()));
                    System.out.print(index++);
                }
            };
//


            final L2 target = (L2) learn.target(LogL2.class);
            final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), target);
            final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), (L2) validate.target(LogL2.class));
            boosting.addListener(counter);
            boosting.addListener(learnListener);
            boosting.addListener(validateListener);
            boosting.fit(learn.vecData(), (L2) learn.target(LogL2.class));
            scores.add(validateListener.min);
        }

        double stat[] = stats(scores);
        System.out.println(String.format("Score stats are %f ± %f, min: %f, max: %f\n\n", stat[0], stat[1], stat[2], stat[3]));
    }


    private static void runBoostingDynamic2(Pool learn, Pool validate, int tries, boolean rf) {
        TDoubleArrayList scores = new TDoubleArrayList();
        for (int tr = 0; tr < tries; ++tr) {
            final GreedyObliviousTreeDynamic2 tree = new GreedyObliviousTreeDynamic2(learn.vecData(), 6, lambda, minBinFactor);
//            tree.stopGrowing();
//            VecOptimization weak =  new BootstrapOptimization(tree, new FastRandom(330));
            VecOptimization weak = rf ? new RandomForest(tree, new FastRandom(random.nextLong()), treesCount)
                    : new BootstrapOptimization(tree, new FastRandom(random.nextLong()));//new RandomForest(tree, random, 4);
//        VecOptimization weak = n

            GradientBoosting<L2> boosting = new GradientBoosting<>(weak, iterations, step);
            final Action counter = new ProgressHandler() {
                int index = 0;

                @Override
                public void invoke(Trans partial) {
                    System.out.println("\n\nCurrent binarization: " + mkString(tree.hist()));
                    System.out.print(index++);
                }
            };
//


            final L2 target = (L2) learn.target(LogL2.class);
            final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), target);
            final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), (L2) validate.target(LogL2.class));
            boosting.addListener(counter);
            boosting.addListener(learnListener);
            boosting.addListener(validateListener);
            boosting.fit(learn.vecData(), (L2) learn.target(LogL2.class));
            scores.add(validateListener.min);
        }

        double stat[] = stats(scores);
        System.out.println(String.format("Score stats are %f ± %f, min: %f, max: %f\n\n", stat[0], stat[1], stat[2], stat[3]));
    }


    private static void runBoostingDynamic3(Pool learn, Pool validate, int tries, boolean rf) {
        TDoubleArrayList scores = new TDoubleArrayList();
        for (int tr = 0; tr < tries; ++tr) {
            final GreedyObliviousTreeDynamic3 tree = new GreedyObliviousTreeDynamic3(learn.vecData(), 6, lambda, minBinFactor);
//            tree.stopGrowing();
//            VecOptimization weak =  new BootstrapOptimization(tree, new FastRandom(330));
            VecOptimization weak = rf ? new RandomForest(tree, new FastRandom(random.nextLong()), treesCount)
                    : new BootstrapOptimization(tree, new FastRandom(random.nextLong()));//new RandomForest(tree, random, 4);
//        VecOptimization weak = n

            GradientBoosting<L2> boosting = new GradientBoosting<>(weak, iterations, step);
            final Action counter = new ProgressHandler() {
                int index = 0;

                @Override
                public void invoke(Trans partial) {
                    System.out.println("\n\nCurrent binarization: " + mkString(tree.hist()));
                    System.out.print(index++);
                }
            };
//


            final L2 target = (L2) learn.target(LogL2.class);
            final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), target);
            final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), (L2) validate.target(LogL2.class));
            boosting.addListener(counter);
            boosting.addListener(learnListener);
            boosting.addListener(validateListener);
            boosting.fit(learn.vecData(), (L2) learn.target(LogL2.class));
            scores.add(validateListener.min);
        }

        double stat[] = stats(scores);
        System.out.println(String.format("Score stats are %f ± %f, min: %f, max: %f\n\n", stat[0], stat[1], stat[2], stat[3]));
    }

    private static void runBoostingDynamicWithRestart(Pool learn, Pool validate, int tries, boolean rf) {
        TDoubleArrayList scores = new TDoubleArrayList();
        for (int tr = 0; tr < tries; ++tr) {
            final GreedyObliviousTreeDynamic tree = new GreedyObliviousTreeDynamic(learn.vecData(), 6, lambda, minBinFactor);
            long seed = random.nextLong();
            VecOptimization weak = rf ? new RandomForest(tree, new FastRandom(seed), treesCount)
                    : new BootstrapOptimization(tree, new FastRandom(seed));//new RandomForest(tree, random, 4);
//        VecOptimization weak = n
            GradientBoosting<L2> boosting = new GradientBoosting<>(weak, iterations, step);
            Action counter = new ProgressHandler() {
                int index = 0;

                @Override
                public void invoke(Trans partial) {
                    System.out.println("\n\nCurrent binarization: " + mkString(tree.hist()));
                    System.out.print(index++);
                }
            };
//

            L2 target = (L2) learn.target(LogL2.class);
            ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), target);
            ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), (L2) validate.target(LogL2.class));
            boosting.addListener(counter);
            boosting.addListener(learnListener);
            boosting.addListener(validateListener);
            boosting.fit(learn.vecData(), (L2) learn.target(LogL2.class));
//            tree.stopGrowing();

//            weak = rf ? new RandomForest(tree, new FastRandom(seed), treesCount)
//                    : new BootstrapOptimization(tree, new FastRandom(seed));//new RandomForest(tree, random, 4);
            boosting = new GradientBoosting<>(weak, aggregateIterations, step);
            counter = new ProgressHandler() {
                int index = 0;

                @Override
                public void invoke(Trans partial) {
                    System.out.println("\n\nCurrent binarization: " + mkString(tree.hist()));
                    System.out.print(index++);
                }
            };
//

            target = (L2) learn.target(LogL2.class);
            learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), target);
            validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), (L2) validate.target(LogL2.class));
            boosting.addListener(counter);
            boosting.addListener(learnListener);
            boosting.addListener(validateListener);
            boosting.fit(learn.vecData(), (L2) learn.target(LogL2.class));
            scores.add(validateListener.min);
        }

        double stat[] = stats(scores);
        System.out.println(String.format("Score stats are %f ± %f, min: %f, max: %f\n\n", stat[0], stat[1], stat[2], stat[3]));
    }


    private static void runBoosting(Pool learn, Pool validate, VecOptimization weak, int tries) {
        TDoubleArrayList scores = new TDoubleArrayList();
        for (int tr = 0; tr < tries; ++tr) {
            GradientBoosting<L2> boosting = new GradientBoosting<>(weak, aggregateIterations, step);
            final Action counter = new ProgressHandler() {
                int index = 0;

                @Override
                public void invoke(Trans partial) {
                    if (partial instanceof Reset) {
                        index = 0;
                        System.out.print("\n\n" + "Boosting was restarted");
                    } else System.out.print("\n" + index++);
                }
            };
//

            final L2 target = (L2) learn.target(LogL2.class);
            final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), target);
            final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), (L2) validate.target(LogL2.class));
            boosting.addListener(counter);
            boosting.addListener(learnListener);
            boosting.addListener(validateListener);
            boosting.fit(learn.vecData(), (L2) learn.target(LogL2.class));
            scores.add(validateListener.min);
        }
        double stat[] = stats(scores);
        System.out.println(String.format("Score stats are %f ± %f, min: %f, max: %f\n\n", stat[0], stat[1], stat[2], stat[3]));
    }


    public static class Reset extends Trans.Stub {

        @Override
        public int xdim() {
            return 0;
        }

        @Override
        public int ydim() {
            return 0;
        }

        @Override
        public Vec trans(Vec x) {
            return null;
        }
    }

    public static class ScoreCalcer implements ProgressHandler {
        final String message;
        Vec current;
        private final VecDataSet ds;
        private final L2 target;

        public ScoreCalcer(String message, VecDataSet ds, L2 target) {
            this.message = message;
            this.ds = ds;
            this.target = target;
            current = new ArrayVec(ds.length());
        }

        public double min = 1e10;

        @Override
        public void invoke(Trans partial) {
            if (partial instanceof Reset) {
                current = new ArrayVec(ds.length());
            } else if (partial instanceof Ensemble) {
                final Ensemble linear;
                linear = (Ensemble) partial;
                final Trans increment = linear.last();
                for (int i = 0; i < ds.length(); i++) {
                    current.adjust(i, linear.wlast() * (increment.trans((ds.data().row(i))).at(0)));
                }
            } else {
                for (int i = 0; i < ds.length(); i++) {
                    current.set(i, ((Func) partial).value(ds.data().row(i)));
                }
            }
            double curLoss = VecTools.distance(current, target.target) / Math.sqrt(ds.length());
            System.out.print(message + curLoss);
            min = Math.min(curLoss, min);
            System.out.print(" minimum = " + min);
        }
    }

    public static String mkString(int[] arr) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < arr.length - 1; ++i) {
            builder.append(arr[i]);
            builder.append(" ");
        }
        builder.append(arr[arr.length - 1]);
        return builder.toString();
    }

    public static String mkString(long[] arr) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < arr.length - 1; ++i) {
            builder.append(arr[i]);
            builder.append(" ");
        }
        builder.append(arr[arr.length - 1]);
        return builder.toString();
    }

    public static <T> String mkString(T[] arr) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < arr.length - 1; ++i) {
            builder.append(arr[i]);
            builder.append(" ");
        }
        builder.append(arr[arr.length - 1]);
        return builder.toString();
    }


}

