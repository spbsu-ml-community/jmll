package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;

import java.io.Serializable;

public interface LambdaStrategyFactory extends Serializable {
    LambdaStrategy get(final TIntObjectMap<Vec> userEmbeddings, final TIntObjectMap<Vec> itemEmbeddings,
                       final TIntDoubleMap initialLambdas, final double beta, final double otherProjectImportance);
}
