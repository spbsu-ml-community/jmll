package com.expleague.erc;

public class Event {
    private final String userId;
    private final String itemId;
    private final double ts;

    public Event(String userId, String itemId, double ts) {
        this.userId = userId;
        this.itemId = itemId;
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

    public boolean isFinish() {
        return true;
    }
}
