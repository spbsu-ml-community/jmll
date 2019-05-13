package com.expleague.erc;

public interface TimeTransformer {
    double apply(double delta, double time, int userId);

    default double apply(Session session) {
        return apply(session.getDelta(), session.getStartTs(), session.userId());
    }
}
