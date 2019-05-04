package com.expleague.erc;

import java.util.ArrayList;
import java.util.List;

public class Session {
    private final List<EventSeq> eventSeqs;

    public Session() {
        eventSeqs = new ArrayList<>();
    }

    public Session(final EventSeq eventSeq) {
        eventSeqs = new ArrayList<>();
        eventSeqs.add(eventSeq);
    }

    public Session(List<EventSeq> eventSeqs) {
        this.eventSeqs = eventSeqs;
    }

    public void add(final EventSeq eventSeq) {
        eventSeqs.add(eventSeq);
    }

    public int userId() {
        return eventSeqs.get(0).userId();
    }

    public double getStartTs() {
        return eventSeqs.get(0).getStartTs();
    }

    public double getDelta() {
        return eventSeqs.get(0).getDelta();
    }

    public List<EventSeq> getEventSeqs() {
        return eventSeqs;
    }
}
