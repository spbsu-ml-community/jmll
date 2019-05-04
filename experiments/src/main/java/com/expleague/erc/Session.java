package com.expleague.erc;

import java.util.ArrayList;
import java.util.List;

public class Session {
    private List<EventSeq> eventSeqs;

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

    public double userId() {
        return eventSeqs.get(0).userId();
    }

    public double startTs() {
        return eventSeqs.get(0).getTs();
    }

    public double delta() {
        return eventSeqs.get(0).getDelta();
    }
}
