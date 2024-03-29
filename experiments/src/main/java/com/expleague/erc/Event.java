package com.expleague.erc;

public class Event {
    protected final int userId;
    protected final int itemId;
    protected final double ts;
    protected double prDelta;

    public Event(int userId, int itemId, double ts) {
        this.userId = userId;
        this.itemId = itemId;
        this.ts = ts;
        prDelta = -1;
    }

    public Event(int userId, int itemId, double ts, double prDelta) {
        this.userId = userId;
        this.itemId = itemId;
        this.ts = ts;
        this.prDelta = prDelta;
    }

    public int userId() {
        return userId;
    }

    public int itemId() {
        return itemId;
    }

    public double getTs() {
        return ts;
    }

    public double getPrDelta() {
        return prDelta;
    }

    public void setPrDelta(final double prDelta) {
        this.prDelta = prDelta;
    }

    public boolean isFinish() {
        return false;
    }

    public long getPair() {
        return Util.combineIds(userId, itemId);
    }
}
