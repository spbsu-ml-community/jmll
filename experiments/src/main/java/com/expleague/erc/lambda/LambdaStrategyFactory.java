package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;

import java.util.Map;

public interface LambdaStrategyFactory {
    LambdaStrategy get(final Map<String, Vec> userEmbeddings, final Map<String, Vec> itemEmbeddings,
                       final double beta, final double otherProjectImportance);
}
