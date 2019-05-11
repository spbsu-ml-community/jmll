package com.expleague.erc.metrics;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.erc.Event;
import com.expleague.erc.models.ApplicableModel;
import com.expleague.erc.models.Model;
import com.expleague.erc.lambda.NotLookAheadLambdaStrategy;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Predicate;

public class LogLikelihoodTest {
    @Test
    public void llTest() {
        final double actualLambda = 1;
        final List<Event> history = Arrays.asList(
                new Event(0, 0, (1 / actualLambda) * 0),
                new Event(0, 0, (1 / actualLambda) * 1),
                new Event(0, 1, (1 / actualLambda) * 1),
                new Event(0, 0, (1 / actualLambda) * 2),
                new Event(0, 1, (1 / actualLambda) * 2)
        );
        final ApplicableModel possibleApplicable1 = new ApplicableMock(.1);
        final ApplicableModel possibleApplicable2 = new ApplicableMock(.01);
        final LogLikelihood llCalculator = new LogLikelihood(.5);
        final double calculatedLL1 = llCalculator.calculate(history, possibleApplicable1);
        final double calculatedLL2 = llCalculator.calculate(history, possibleApplicable2);
        Assert.assertTrue(calculatedLL1 > calculatedLL2);
    }

    @Test
    public void diffTest() {
        final List<Event> differentiationHistory = Arrays.asList(
                new Event(0, 0, 0),
                new Event(1, 0, 1),
                new Event(0, 1, 2),
                new Event(0, 0, 3, 3),
                new Event(0, 1, 4, 2),
                new Event(1, 0, 5, 4),
                new Event(0, 1, 6, 2),
                new Event(1, 1, 7),
                new Event(1, 1, 8, 1),
                new Event(1, 0, 9, 4)
        );
        final List<Event> preparationHistory = Collections.emptyList();

        final TIntObjectMap<Vec> usersEmbeddings = new TIntObjectHashMap<>();
        usersEmbeddings.put(0, new ArrayVec(.16, .18, .2, .22, .24));
        usersEmbeddings.put(1, new ArrayVec(.18, .19, .2, .21, .22));
//        final TIntDoubleMap usersBiases = new TIntDoubleHashMap();
//        usersBiases.put(0, 0);
//        usersBiases.put(1, 0);
        final TIntObjectMap<Vec> itemsEmbeddings = new TIntObjectHashMap<>();
        itemsEmbeddings.put(0, new ArrayVec(.16, .18, .2, .22, .24));
        itemsEmbeddings.put(1, new ArrayVec(.15, .18, .2, .22, .25));
//        final TIntDoubleMap itemBiases = new TIntDoubleHashMap();
//        itemBiases.put(0, 0);
//        itemBiases.put(1, 0);

        final Model model = new Model(5, 0.1, 5, 0.1,
                x -> x, x -> 1, new NotLookAheadLambdaStrategy.NotLookAheadLambdaStrategyFactory(),
                usersEmbeddings, itemsEmbeddings);

        final LogLikelihood llCalculator = new LogLikelihood(5);

        final TIntObjectMap<Vec> usersNumericDerivative = new TIntObjectHashMap<>();
        final TIntObjectMap<Vec> itemsNumericDerivative = new TIntObjectHashMap<>();
        final TIntObjectMap<Vec> usersAnalyticDerivative = new TIntObjectHashMap<>();
        final TIntObjectMap<Vec> itemsAnalyticDerivative = new TIntObjectHashMap<>();
        double step = 1e-6;
        LogLikelihood.differentiate(usersEmbeddings, usersNumericDerivative, model, llCalculator,
                preparationHistory, differentiationHistory, step);
        LogLikelihood.differentiate(itemsEmbeddings, itemsNumericDerivative, model, llCalculator,
                preparationHistory, differentiationHistory, step);
        model.logLikelihoodDerivative(differentiationHistory, usersAnalyticDerivative, itemsAnalyticDerivative, null);

        final double maxDist = 1.;
        usersEmbeddings.forEachKey(userId -> {
            System.out.println(usersNumericDerivative.get(userId));
            System.out.println(usersAnalyticDerivative.get(userId));
            final double dist = VecTools.distance(usersNumericDerivative.get(userId), usersAnalyticDerivative.get(userId));
            System.out.println(dist);
            Assert.assertTrue(dist < maxDist);
            return true;
        });
        System.out.println();
        itemsEmbeddings.forEachKey(itemId -> {
            System.out.println(itemsNumericDerivative.get(itemId));
            System.out.println(itemsAnalyticDerivative.get(itemId));
            final double dist = VecTools.distance(itemsNumericDerivative.get(itemId), itemsAnalyticDerivative.get(itemId));
            System.out.println(dist);
            Assert.assertTrue(dist < maxDist);
            return true;
        });
    }
}
