package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.erc.Event;

import java.util.Map;

public interface LambdaStrategy {
    double getLambda(String userId, String itemId);

    ArrayVec getLambdaUserDerivative(String userId, String itemId);

    Map<String, ArrayVec> getLambdaProjectDerivative(String userId, String itemId);

    void accept(Event event);

}
