package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Model;

public class ApplicableMock implements Model.Applicable {
    private final double lambda;

    public ApplicableMock(double lambda) {
        this.lambda = lambda;
    }

    @Override
    public void accept(Event event) {

    }

    @Override
    public double getLambda(int userId, int itemId) {
        return lambda;
    }

    @Override
    public double timeDelta(int userId, int itemId) {
        return 1 / lambda;
    }
}
