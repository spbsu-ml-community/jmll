package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.models.ApplicableModel;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

public class SPUTest {
    @Test
    public void accurateTest() {
        final double lambda = 1;
        final List<Event> history = Arrays.asList(
                new Event(0, 0, (1 / lambda) * 0),
                new Event(0, 0, (1 / lambda) * 1),
                new Event(0, 1, (1 / lambda) * 1),
                new Event(0, 0, (1 / lambda) * 2),
                new Event(0, 1, (1 / lambda) * 2)
        );
        final ApplicableModel applicable = new ApplicableMock(lambda);
        final double calculatedSPU = new SPU().calculate(history, applicable);
        Assert.assertEquals(0, calculatedSPU, 1e-9);
    }

    @Test
    public void inaccurateTest() {
        final double actualLambda = 1;
        final List<Event> history = Arrays.asList(
                new Event(0, 0, (1 / actualLambda) * 0),
                new Event(0, 0, (1 / actualLambda) * 1),
                new Event(0, 1, (1 / actualLambda) * 1),
                new Event(0, 0, (1 / actualLambda) * 2),
                new Event(0, 1, (1 / actualLambda) * 2)
        );
        final ApplicableModel possibleApplicable1 = new ApplicableMock(2);
        final ApplicableModel possibleApplicable2 = new ApplicableMock(3);
        final SPU spuCalculator = new SPU();
        final double calculatedSPU1 = spuCalculator.calculate(history, possibleApplicable1);
        final double calculatedSPU2 = spuCalculator.calculate(history, possibleApplicable2);
        Assert.assertTrue(calculatedSPU1 < calculatedSPU2);
    }
}
