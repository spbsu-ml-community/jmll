package com.expleague.erc.models;

import com.expleague.erc.Event;

import java.util.List;

public interface ApplicableModel {
    void accept(final Event event);

    double getLambda(final int userId, final int itemId);

    double timeDelta(final int userId, final int itemId);

    double probabilityBeforeX(final int userId, final int itemId, final double x);

    default double probabilityInterval(final int userId, final int itemId, final double start, final double end) {
        return probabilityBeforeX(userId, itemId, end) - probabilityBeforeX(userId, itemId, start);
    }

    default ApplicableModel fit(final List<Event> history) {
        for (Event event : history) {
            accept(event);
        }
        return this;
    }
}
