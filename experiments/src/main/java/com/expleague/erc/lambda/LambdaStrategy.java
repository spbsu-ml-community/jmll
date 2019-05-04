package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.EventSeq;
import gnu.trove.map.TIntObjectMap;

import java.io.Serializable;

public interface LambdaStrategy extends Serializable {
    double getLambda(final int userId, final int itemId);

    double getLambda(final int userId);

    Vec getLambdaUserDerivative(final int userId);

    Vec getLambdaUserDerivative(final int userId, final int itemId);

    TIntObjectMap<Vec> getLambdaItemDerivative(final int userId);

    TIntObjectMap<Vec> getLambdaItemDerivative(final int userId, final int itemId);

    void accept(final EventSeq eventSeq);
}
