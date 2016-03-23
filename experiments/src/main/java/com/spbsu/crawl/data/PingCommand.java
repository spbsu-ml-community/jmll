package com.spbsu.crawl.data;

import java.util.concurrent.BlockingQueue;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */
public class PingCommand implements Command {
  @Override
  public void execute(BlockingQueue<Message> responseQueue) {
    responseQueue.add(new PongMessage());
  }
}
