package com.expleague.erc;

public class Event {
    private final String userId;
    private final String itemId;
    private final double ts;

    public Event(String uid, String pid, double ts) {
        this.userId = uid;
        this.itemId = pid;
        this.ts = ts;
    }

    public String userId() {
        return userId;
    }

    public String itemId() {
        return itemId;
    }

    public double getTs() {
        return ts;
    }

    public Double getPrDelta() {
        throw new UnsupportedOperationException();
    }

    public int getNTasks() {
        throw new UnsupportedOperationException();
    }
}
