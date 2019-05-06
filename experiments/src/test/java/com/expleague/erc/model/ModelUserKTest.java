package com.expleague.erc.model;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.Event;
import com.expleague.erc.lambda.NotLookAheadLambdaStrategy;
import com.expleague.erc.metrics.MAEPerPair;
import com.expleague.erc.models.ApplicableModel;
import com.expleague.erc.models.ModelUserK;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import static java.lang.Math.*;

import static com.expleague.erc.GammaUtils.*;

public class ModelUserKTest {
    @Test
    public void paramsEstimationTest() {
        final List<Event> history = Arrays.asList(
                new Event(0, 0, 0),
                new Event(0, 0, 1, 1),
                new Event(0, 0, 11, 10),
                new Event(0, 0, 111, 100),
                new Event(0, 0, 1111, 1000),
                new Event(1, 0, 0),
                new Event(1, 0, 10, 10),
                new Event(1, 0, 30, 20),
                new Event(1, 0, 60, 30),
                new Event(1, 0, 100, 40)
        );
        history.sort(Comparator.comparingDouble(Event::getTs));
        final TIntDoubleMap userBaseLambdas = new TIntDoubleHashMap();
        final TIntIntMap userKs = new TIntIntHashMap();
        ModelUserK.calcUserParams(history, userBaseLambdas, userKs);
        Assert.assertEquals(2, userBaseLambdas.size());
        Assert.assertEquals(2, userKs.size());
        Assert.assertEquals(0.0015837, userBaseLambdas.get(0), 1e-6);
        Assert.assertEquals(0.2, userBaseLambdas.get(1), 1e-6);
    }

    @Test
    public void singleUserTest() {
        final List<Event> history = Arrays.asList(
                new Event(0, 0, 0),
                new Event(0, 0, 10, 10),
                new Event(0, 0, 30, 20),
                new Event(0, 0, 40, 10),
                new Event(0, 0, 60, 20),
                new Event(0, 0, 70, 10),
                new Event(0, 0, 90, 20)
        );
        final TIntDoubleMap userBaseLambdas = new TIntDoubleHashMap();
        final TIntIntMap userKs = new TIntIntHashMap();
        ModelUserK.calcUserParams(history, userBaseLambdas, userKs);
        final TIntObjectMap<Vec> userEmbeddings = new TIntObjectHashMap<>();
        final TIntObjectMap<Vec> itemEmbeddings = new TIntObjectHashMap<>();
        ModelUserK.makeInitialEmbeddings(3, userBaseLambdas, history, userEmbeddings, itemEmbeddings);
        final ModelUserK model = new ModelUserK(3, 0.1, 0.5, 0.1,
                Math::abs, Math::signum, new NotLookAheadLambdaStrategy.NotLookAheadLambdaStrategyFactory(),
                userEmbeddings, itemEmbeddings, userKs, userBaseLambdas);
        model.fit(history, 1e-2, 100, 0.99, m -> {});
        Assert.assertTrue(model.getApplicable().timeDelta(0, 0) > 15);
        Assert.assertTrue(model.getApplicable(history).timeDelta(0, 0) > 15);
        Assert.assertTrue(new MAEPerPair().calculate(history, model.getApplicable()) < 5);
    }

    @Test
    public void doubleUserTest() {
        final List<Event> history = Arrays.asList(
                new Event(0, 0, 0),
                new Event(1, 0, 0),
                new Event(0, 0, 1, 1),
                new Event(0, 0, 3, 2),
                new Event(0, 0, 6, 3),
                new Event(0, 0, 10, 4),
                new Event(1, 0, 10, 10),
                new Event(1, 0, 30, 20),
                new Event(1, 0, 60, 30),
                new Event(1, 0, 100, 40)
        );
        final TIntDoubleMap userBaseLambdas = new TIntDoubleHashMap();
        final TIntIntMap userKs = new TIntIntHashMap();
        ModelUserK.calcUserParams(history, userBaseLambdas, userKs);
        final TIntObjectMap<Vec> userEmbeddings = new TIntObjectHashMap<>();
        final TIntObjectMap<Vec> itemEmbeddings = new TIntObjectHashMap<>();
        ModelUserK.makeInitialEmbeddings(3, userBaseLambdas, history, userEmbeddings, itemEmbeddings);
        final ModelUserK model = new ModelUserK(3, 0.1, 0.5, 0.1,
                Math::abs, Math::signum, new NotLookAheadLambdaStrategy.NotLookAheadLambdaStrategyFactory(),
                userEmbeddings, itemEmbeddings, userKs, userBaseLambdas);
        model.fit(history, 1e-2, 100, 0.99, m -> {});
        final ApplicableModel applicable = model.getApplicable();
        Assert.assertEquals(10, applicable.timeDelta(1, 0) / applicable.timeDelta(0, 0), 1);
    }

    @Test
    public void paramsRestorationTest() {
        final int K = 3;
        final double THETA = 1e-3;
        final List<Event> history = new ArrayList<>();
        history.add(new Event(0, 0, 0));
        double lastTime = 0;
        for (int i = 0; i < 200; ++i) {
            final double timeDelta = genGamma(K, THETA);
            lastTime += timeDelta;
            history.add(new Event(0, 0, lastTime, timeDelta));
        }
        final TIntDoubleMap userBaseLambdas = new TIntDoubleHashMap();
        final TIntIntMap userKs = new TIntIntHashMap();
        ModelUserK.calcUserParams(history, userBaseLambdas, userKs);
        Assert.assertEquals(K, userKs.get(0));
        Assert.assertEquals(1, THETA / userBaseLambdas.get(0), .25);

        final TIntObjectMap<Vec> userEmbeddings = new TIntObjectHashMap<>();
        final TIntObjectMap<Vec> itemEmbeddings = new TIntObjectHashMap<>();
        ModelUserK.makeInitialEmbeddings(3, userBaseLambdas, history, userEmbeddings, itemEmbeddings);
        final ModelUserK model = new ModelUserK(3, 0.1, 0.5, 0.1,
                Math::abs, Math::signum, new NotLookAheadLambdaStrategy.NotLookAheadLambdaStrategyFactory(),
                userEmbeddings, itemEmbeddings, userKs, userBaseLambdas);
//        System.out.println(model.getApplicable().getLambda(0, 0) + " " + model.getApplicable(history).getLambda(0, 0));
        model.fit(history, 1e-5, 1000, 0.99, m -> {});
//        System.out.println(model.getApplicable().getLambda(0, 0) + " " + model.getApplicable(history).getLambda(0, 0));
//        System.out.println(model.getApplicable(history).timeDelta(0, 0));
        Assert.assertEquals(1, THETA / model.getApplicable(history).getLambda(0, 0), .1);
    }

    private double EPS = 1e-6;
    private double genGamma(int k, double theta) {
        final double fVal = random();
        double left = 0, right = 1e9;
        while (right - left > EPS) {
            final double mid = (left + right) / 2;
            if (regularizedGammaP(k, mid * theta) > fVal) {
                right = mid;
            } else {
                left = mid;
            }
        }
        return left;
    }
}
