package com.expleague.erc;

import com.expleague.erc.models.ModelDays;

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

    public boolean isStart() {
        return getDelta() == -1;
    }

//    public double toDayTs(double time, int userBorder, final int dayLength) {
//        double lastBorder = ((int) time / dayLength) * dayLength + userBorder;
//        if (lastBorder > time) {
//            lastBorder -= dayLength;
//        }
//        return lastBorder;
//    }
//
//    public int getDayDelta(final int userBorder, final int dayLength) {
//        return (int) (toDayTs(getStartTs(), userBorder, dayLength) -
//                toDayTs(getStartTs() - getDelta(), userBorder, dayLength)) / dayLength;
//    }

    public List<EventSeq> getEventSeqs() {
        return eventSeqs;
    }
}
