package com.expleague.erc;

public class Event {
    private final String uid;
    private final String pid;
    private final double ts;

    public Event(String uid, String pid, double ts) {
        this.uid = uid;
        this.pid = pid;
        this.ts = ts;
    }

    public String getUid() {
        return uid;
    }

    public String getPid() {
        return pid;
    }

    public double getTs() {
        return ts;
    }

    public Integer getPrDelta() {
        throw new UnsupportedOperationException();
    }

    public int getNTasks() {
        throw new UnsupportedOperationException();
    }
}
