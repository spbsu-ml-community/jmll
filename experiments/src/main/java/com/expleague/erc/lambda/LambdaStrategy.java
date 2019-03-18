package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.Event;

import java.util.Map;

public interface LambdaStrategy {
    double getLambda(final String userId, final String itemId);

    Vec getLambdaUserDerivative(final String userId, final String itemId);

    Map<String, Vec> getLambdaItemDerivative(final String userId, final String itemId);

    void accept(final Event event);
}
