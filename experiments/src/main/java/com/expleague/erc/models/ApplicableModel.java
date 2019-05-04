package com.expleague.erc.models;

import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Session;
import com.expleague.erc.data.DataPreprocessor;

import java.util.List;

public interface ApplicableModel {
    void accept(final EventSeq event);

    default void accept(final Session session) {
        session.getEventSeqs().forEach(this::accept);
    }

    double getLambda(final int userId);

    double getLambda(final int userId, final int itemId);

    double timeDelta(final int userId, final int itemId);

    double timeDelta(final int userId);

    double probabilityBeforeX(final int userId, final double x);

    double probabilityBeforeX(final int userId, final int itemId, final double x);

    default double probabilityInterval(final int userId, final double start, final double end) {
        return probabilityBeforeX(userId, end) - probabilityBeforeX(userId, start);
    }

    default double probabilityInterval(final int userId, final int itemId, final double start, final double end) {
        return probabilityBeforeX(userId, itemId, end) - probabilityBeforeX(userId, itemId, start);
    }

    default ApplicableModel fit(final List<Event> history) {
        for (final EventSeq eventSeq : DataPreprocessor.groupToEventSeqs(history)) {
            accept(eventSeq);
        }
        return this;
    }
}
