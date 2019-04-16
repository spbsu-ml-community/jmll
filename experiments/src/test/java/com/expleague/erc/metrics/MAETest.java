package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Metrics.MAE;
import com.expleague.erc.Model;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

public class MAETest {
    @Test
    public void basicTest() {
        final double lambda = 1;
        final Model.Applicable applicableMock = new ApplicableMock(lambda);
        final List<Event> history = Arrays.asList(
                new Event(0, 0, 0, -1),
                new Event(0, 0, 1, 1)
        );
        final double calculatedMAE = new MAE().calculate(history, applicableMock);
        final double expectedMAE = Math.abs(1 - 1 / lambda);
        Assert.assertEquals(expectedMAE, calculatedMAE, 1e-9);
    }

    @Test
    public void complicatedTest() {
        final double lambda = 2;
        final Model.Applicable applicableMock = new ApplicableMock(lambda);
        final List<Event> history = Arrays.asList(
                new Event(0, 0, 0, -1),
                new Event(0, 0, 1, 1),
                new Event(0, 1, 2, -1),
                new Event(1, 0, 3, -1),
                new Event(0, 1, 4, 2),
                new Event(1, 0, 5, 2)
        );
        final double calculatedMAE = new MAE().calculate(history, applicableMock);
        final double expectedMAE = (0.5 + 1.5 + 1.5) / 3;
        Assert.assertEquals(expectedMAE, calculatedMAE, 1e-9);
    }
}
